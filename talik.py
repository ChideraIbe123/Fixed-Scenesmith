"""Talik — Web UI for SceneSmith scene generation.

Single-file FastAPI backend that serves the Talik frontend, manages pipeline
execution as an async subprocess, and serves the resulting GLB for 3D viewing.

Usage:
    cd /workspace/scenesmith && python talik.py
"""

import asyncio
import logging
import os
import re
import shutil
import subprocess
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import uvicorn

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.requests import Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("talik")

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
VENV_PYTHON = str(BASE_DIR / ".venv" / "bin" / "python")

# Stage definitions in pipeline order
STAGES = [
    ("floor_plan", "Floor Plan"),
    ("furniture", "Furniture"),
    ("wall", "Walls"),
    ("ceiling", "Ceiling"),
    ("manipuland", "Manipulands"),
    ("done", "Done"),
]

STAGE_PATTERNS = [
    (re.compile(r"Generating house layout"), "floor_plan"),
    (re.compile(r"Adding furniture"), "furniture"),
    (re.compile(r"wall_objects|Adding wall objects|wall object"), "wall"),
    (re.compile(r"Adding ceiling|ceiling objects"), "ceiling"),
    (re.compile(r"[Mm]anipuland"), "manipuland"),
    (re.compile(r"Experiment execution completed"), "done"),
]


@dataclass
class Job:
    prompt: str
    status: str = "running"  # running | done | error
    stage: str = "floor_plan"
    start_time: float = field(default_factory=time.time)
    process: Optional[asyncio.subprocess.Process] = None
    output_dir: Optional[Path] = None
    scene_dir: Optional[Path] = None
    glb_path: Optional[Path] = None
    log_tail: str = ""
    error: str = ""


app = FastAPI(title="Talik")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Single-user, one job at a time
current_job: Optional[Job] = None


def detect_stage(log_text: str) -> str:
    """Detect pipeline stage from log content. Returns the latest stage found."""
    detected = "floor_plan"
    for pattern, stage in STAGE_PATTERNS:
        if pattern.search(log_text):
            detected = stage
    return detected


def find_latest_preview(scene_dir: Path) -> Optional[Path]:
    """Find the most recent rendered PNG in the scene directory."""
    if not scene_dir or not scene_dir.exists():
        return None
    # Look in room_*/scene_renders/*/renders_*/0_side.png
    renders = sorted(scene_dir.glob("room_*/scene_renders/*/renders_*/0_side.png"))
    if renders:
        return renders[-1]
    # Fallback: any PNG
    pngs = sorted(scene_dir.glob("room_*/scene_renders/**/*.png"))
    return pngs[-1] if pngs else None


def find_scene_log(job: Job) -> Optional[Path]:
    """Find the scene.log for the current job."""
    if job.scene_dir and (job.scene_dir / "scene.log").exists():
        return job.scene_dir / "scene.log"
    if job.output_dir and (job.output_dir / "experiment.log").exists():
        return job.output_dir / "experiment.log"
    return None


def read_log_tail(job: Job, lines: int = 30) -> str:
    """Read the last N lines of the job's log."""
    log_path = find_scene_log(job)
    if not log_path or not log_path.exists():
        return ""
    try:
        text = log_path.read_text(errors="replace")
        return "\n".join(text.splitlines()[-lines:])
    except Exception:
        return ""


def resolve_output_dir() -> Optional[Path]:
    """Resolve the current run's output directory via the latest-run symlink."""
    symlink = OUTPUTS_DIR / "latest-run"
    if symlink.exists():
        return symlink.resolve()
    return None


async def run_pipeline(job: Job):
    """Launch main.py as subprocess, monitor progress, convert to GLB on completion."""
    global current_job
    try:
        env = {
            **os.environ,
            "MUJOCO_GL": "egl",
        }
        # Forward relevant env vars
        for key in ("OPENAI_API_KEY", "AZURE_HSSD_CONNECTION_STRING", "OPENAI_TRACING_KEY"):
            val = os.environ.get(key)
            if val:
                env[key] = val

        cmd = [
            VENV_PYTHON, "main.py",
            "+name=talik_run",
            f'experiment.prompts=["{job.prompt}"]',
            "furniture_agent.asset_manager.general_asset_source=hssd",
            "wall_agent.asset_manager.general_asset_source=hssd",
            "ceiling_agent.asset_manager.general_asset_source=hssd",
            "manipuland_agent.asset_manager.general_asset_source=hssd",
        ]

        # Snapshot the current latest-run target so we can detect when it changes
        old_target = resolve_output_dir()
        logger.info(f"Starting pipeline: {' '.join(cmd)}")
        logger.info(f"Previous latest-run target: {old_target}")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(BASE_DIR),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        job.process = proc

        # Monitor the process
        while True:
            # Check if process ended
            if proc.returncode is not None:
                break

            # Try to discover output dir — only accept if symlink changed
            if not job.output_dir:
                resolved = resolve_output_dir()
                if resolved and resolved != old_target:
                    job.output_dir = resolved
                    job.scene_dir = resolved / "scene_000"
                    logger.info(f"Discovered NEW output dir: {resolved}")

            # Update log tail and stage
            tail = read_log_tail(job)
            if tail:
                job.log_tail = tail
                # Read full log for stage detection
                log_path = find_scene_log(job)
                if log_path and log_path.exists():
                    try:
                        full_log = log_path.read_text(errors="replace")
                        job.stage = detect_stage(full_log)
                    except Exception:
                        pass

            # Wait before next check
            try:
                await asyncio.wait_for(proc.wait(), timeout=3.0)
                break
            except asyncio.TimeoutError:
                continue

        # Process finished
        await proc.wait()

        # Final output dir resolution
        if not job.output_dir:
            resolved = resolve_output_dir()
            if resolved and resolved != old_target:
                job.output_dir = resolved
                job.scene_dir = resolved / "scene_000"

        if proc.returncode != 0:
            # Read any remaining output
            remaining = await proc.stdout.read()
            job.log_tail = read_log_tail(job) or remaining.decode(errors="replace")[-2000:]
            job.status = "error"
            job.error = f"Pipeline exited with code {proc.returncode}"
            logger.error(f"Pipeline failed: exit code {proc.returncode}")
            return

        logger.info("Pipeline completed successfully, converting to GLB...")
        job.stage = "done"
        job.log_tail = read_log_tail(job)

        # Find house.blend and convert to GLB
        glb_ok = await convert_to_glb(job)
        if glb_ok:
            job.status = "done"
            logger.info(f"GLB ready at: {job.glb_path}")
        else:
            job.status = "error"
            job.error = "GLB conversion failed — scene was generated but 3D export failed"
            logger.error("GLB conversion failed")

    except Exception as e:
        logger.exception("Pipeline error")
        job.status = "error"
        job.error = str(e)


async def convert_to_glb(job: Job) -> bool:
    """Convert house.blend to scene.glb using Blender subprocess."""
    if not job.scene_dir:
        return False

    # Find house.blend — prefer combined_house, then combined_house_after_*
    blend_candidates = [
        job.scene_dir / "combined_house" / "house.blend",
        job.scene_dir / "combined_house_after_ceiling" / "house.blend",
        job.scene_dir / "combined_house_after_wall_objects" / "house.blend",
        job.scene_dir / "combined_house_after_furniture" / "house.blend",
    ]
    blend_path = None
    for candidate in blend_candidates:
        if candidate.exists():
            blend_path = candidate
            break

    if not blend_path:
        # Search more broadly
        blends = sorted(job.scene_dir.glob("**/house.blend"))
        if blends:
            blend_path = blends[-1]

    if not blend_path:
        logger.error(f"No house.blend found in {job.scene_dir}")
        return False

    glb_path = job.scene_dir / "scene.glb"
    logger.info(f"Converting {blend_path} → {glb_path}")

    blender_script = f"""
import bpy
bpy.ops.wm.open_mainfile(filepath=r"{blend_path}")
bpy.ops.export_scene.gltf(filepath=r"{str(glb_path)}", export_format="GLB", export_yup=True)
print("GLB export complete")
"""
    # Run Blender headless via the venv python (which has bpy)
    proc = await asyncio.create_subprocess_exec(
        VENV_PYTHON, "-c", blender_script,
        cwd=str(BASE_DIR),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    output = stdout.decode(errors="replace")

    if proc.returncode != 0 or not glb_path.exists():
        logger.error(f"Blender GLB export failed (rc={proc.returncode}): {output[-500:]}")
        return False

    job.glb_path = glb_path
    return True


# --- Routes ---


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("talik.html", {"request": request})


@app.get("/tc-test", response_class=HTMLResponse)
async def tc_test(request: Request):
    return templates.TemplateResponse("tc_test.html", {"request": request})


@app.post("/api/generate")
async def generate(request: Request):
    global current_job

    if current_job and current_job.status == "running":
        raise HTTPException(status_code=409, detail="A generation is already in progress")

    body = await request.json()
    prompt = body.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    job = Job(prompt=prompt)
    current_job = job

    # Launch pipeline in background
    asyncio.create_task(run_pipeline(job))

    return JSONResponse({"status": "started", "prompt": prompt})


@app.get("/api/status")
async def status():
    if not current_job:
        return JSONResponse({
            "status": "idle",
            "stage": None,
            "elapsed": 0,
            "log_tail": "",
            "preview_url": None,
            "error": None,
        })

    job = current_job
    elapsed = int(time.time() - job.start_time)

    # Find preview image
    preview_url = None
    if job.scene_dir:
        preview = find_latest_preview(job.scene_dir)
        if preview:
            preview_url = "/api/preview"

    return JSONResponse({
        "status": job.status,
        "stage": job.stage,
        "elapsed": elapsed,
        "log_tail": job.log_tail[-2000:] if job.log_tail else "",
        "preview_url": preview_url,
        "error": job.error or None,
    })


@app.post("/api/cancel")
async def cancel():
    global current_job
    if not current_job or current_job.status != "running":
        raise HTTPException(status_code=400, detail="No running job to cancel")

    job = current_job
    if job.process and job.process.returncode is None:
        import signal
        try:
            os.killpg(os.getpgid(job.process.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            try:
                job.process.terminate()
            except ProcessLookupError:
                pass
    job.status = "error"
    job.error = "Cancelled by user"
    logger.info("Job cancelled by user")
    return JSONResponse({"status": "cancelled"})


@app.get("/api/scene.glb")
async def serve_glb():
    if not current_job or not current_job.glb_path or not current_job.glb_path.exists():
        raise HTTPException(status_code=404, detail="GLB not available yet")
    return FileResponse(
        current_job.glb_path,
        media_type="model/gltf-binary",
        filename="scene.glb",
    )


@app.get("/api/preview")
async def serve_preview():
    if not current_job or not current_job.scene_dir:
        raise HTTPException(status_code=404, detail="No preview available")
    preview = find_latest_preview(current_job.scene_dir)
    if not preview:
        raise HTTPException(status_code=404, detail="No preview available")
    return FileResponse(preview, media_type="image/png")


@app.post("/api/load-existing")
async def load_existing():
    """Debug: load the latest existing scene run for 3D viewer testing."""
    global current_job

    resolved = resolve_output_dir()
    if not resolved:
        raise HTTPException(status_code=404, detail="No existing run found in outputs/latest-run")

    scene_dir = resolved / "scene_000"

    # If scene_000 doesn't have a house.blend, search all outputs
    if not list(scene_dir.glob("**/house.blend")):
        logger.warning(f"No house.blend in {scene_dir}, searching all outputs...")
        all_blends = sorted(OUTPUTS_DIR.glob("**/combined_house/house.blend"))
        if all_blends:
            scene_dir = all_blends[-1].parent.parent
            resolved = scene_dir.parent
            logger.info(f"Found house.blend in: {scene_dir}")
        else:
            raise HTTPException(status_code=404, detail="No completed scene found")

    glb_path = scene_dir / "scene.glb"

    # If GLB doesn't exist yet, try to convert
    if not glb_path.exists():
        job = Job(prompt="(loaded existing scene)", status="running")
        job.output_dir = resolved
        job.scene_dir = scene_dir
        current_job = job
        ok = await convert_to_glb(job)
        if not ok:
            current_job = None
            raise HTTPException(status_code=500, detail="GLB conversion failed")

    job = Job(prompt="(loaded existing scene)", status="done")
    job.output_dir = resolved
    job.scene_dir = scene_dir
    job.glb_path = glb_path
    job.stage = "done"
    current_job = job

    return JSONResponse({"status": "done", "glb_size_mb": round(glb_path.stat().st_size / 1e6, 1)})


ROBOT_UPLOADS_DIR = BASE_DIR / "outputs" / "robot_uploads"


@app.post("/api/upload-robot")
async def upload_robot(
    urdf: UploadFile = File(...),
    stl_files: list[UploadFile] = File(...),
):
    """Upload URDF + STL files. Saves them to a served directory and rewrites
    package:// URIs so the browser-side urdf-loader can fetch them."""

    robot_name = Path(urdf.filename).stem
    robot_dir = ROBOT_UPLOADS_DIR / robot_name
    meshes_dir = robot_dir / "meshes"

    # Clean previous upload of same robot
    if robot_dir.exists():
        shutil.rmtree(robot_dir)
    robot_dir.mkdir(parents=True)
    meshes_dir.mkdir()

    try:
        # Save STL files flat into meshes/
        stl_names = []
        for stl in stl_files:
            stl_path = meshes_dir / stl.filename
            stl_content = await stl.read()
            stl_path.write_bytes(stl_content)
            stl_names.append(stl.filename)
            logger.info(f"Saved STL: {stl_path} ({len(stl_content)} bytes)")

        # Save and rewrite URDF — replace package:// mesh URIs with relative
        # paths like ./meshes/filename.stl so urdf-loader can fetch them.
        urdf_text = (await urdf.read()).decode("utf-8")

        # Build lookup: lowercase basename -> actual filename
        stl_lookup = {name.lower(): name for name in stl_names}

        def _rewrite_mesh_uri(match):
            original = match.group(1)
            basename = Path(original).name.lower()
            if basename in stl_lookup:
                return f'filename="meshes/{stl_lookup[basename]}"'
            logger.warning(f"No uploaded mesh matches: {original}")
            return match.group(0)

        urdf_text = re.sub(r'filename="([^"]+)"', _rewrite_mesh_uri, urdf_text)
        urdf_path = robot_dir / urdf.filename
        urdf_path.write_text(urdf_text)
        logger.info(f"Saved URDF: {urdf_path} (rewrote {len(stl_lookup)} mesh paths)")

        # Return the URL path the frontend will use to load via urdf-loader
        urdf_url = f"/robot-files/{robot_name}/{urdf.filename}"
        return JSONResponse({
            "status": "done",
            "robot_name": robot_name,
            "urdf_url": urdf_url,
            "meshes": stl_names,
        })

    except Exception as e:
        logger.exception("Robot upload failed")
        raise HTTPException(status_code=500, detail=str(e))


# Static files: serve renders/, outputs/, and robot uploads.
# Must be mounted AFTER API routes.
ROBOT_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/robot-files", StaticFiles(directory=str(ROBOT_UPLOADS_DIR)), name="robot-files")
if (BASE_DIR / "renders").exists():
    app.mount("/renders", StaticFiles(directory=str(BASE_DIR / "renders")), name="renders")
if (BASE_DIR / "outputs").exists():
    app.mount("/outputs", StaticFiles(directory=str(BASE_DIR / "outputs")), name="outputs")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="info")
