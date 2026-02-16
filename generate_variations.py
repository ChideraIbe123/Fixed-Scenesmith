#!/usr/bin/env python3
import argparse, json, os, subprocess, sys
from datetime import datetime
from pathlib import Path
from openai import OpenAI

BASE_PROMPT = (
    "A modern kitchen room with an island counter, stainless steel appliances, "
    "wooden cabinets, a sink, and a dining table with four chairs"
)

def generate_manipuland_variations(num_variations=3):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """You are a scene design expert specializing in small object placement in kitchen environments. The kitchen's layout, walls, floor, and major furniture are ALREADY PLACED and will NOT change. Generate variations of what SMALL OBJECTS should be placed on existing kitchen surfaces (counters, island, table, shelves). Items like utensils, plates, cups, food items, small appliances, cutting boards, spice jars, cookbooks, potted herbs, towels, etc. Each variation should represent a different moment or state: morning breakfast vs evening dinner, clean minimalist vs busy cooking, baking session vs coffee morning vs entertaining guests. Return ONLY a JSON array of strings."""},
            {"role": "user", "content": f'Kitchen: "{BASE_PROMPT}"\n\nGenerate {num_variations} variations of small object arrangements. Each should be 1-2 sentences. Return as a JSON array of strings.'}
        ],
        temperature=0.9,
    )
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        content = content.rsplit("```", 1)[0]
    return json.loads(content)

def run_scenesmith(cmd_args, label=""):
    cmd = [sys.executable, "main.py"] + cmd_args
    print(f"\n{'='*60}\n  {label}\n{'='*60}\n")
    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()
    if process.returncode != 0:
        print(f"\nERROR: Exited with code {process.returncode}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Generate kitchen scene variations")
    parser.add_argument("-n", "--num-variations", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set"); sys.exit(1)

    print(f"Base prompt: {BASE_PROMPT}\nVariations: {args.num_variations}\n")
    print("Generating manipuland variation prompts...")
    variations = generate_manipuland_variations(args.num_variations)

    print("\nVariation prompts:\n" + "-"*60)
    for i, v in enumerate(variations):
        print(f"  [{i}] {v}")
    print("-"*60 + "\n")

    if args.dry_run:
        print("[DRY RUN] Done."); return

    timestamp = datetime.now().strftime("%H%M%S")
    prompt_json = json.dumps([BASE_PROMPT])

    print("STEP 1/2: Generating base kitchen (floor plan + furniture)...")
    success = run_scenesmith([
        f"+name=kitchen_base_{timestamp}",
        f"experiment.prompts={prompt_json}",
        "floor_plan_agent.mode=room",
        "experiment.pipeline.stop_stage=furniture",
    ], label="Base kitchen: floor plan + furniture (30-60 min)")

    if not success:
        print("Base scene failed!"); sys.exit(1)

    today = datetime.now().strftime("%Y-%m-%d")
    today_dir = Path("outputs") / today
    base_path = sorted(today_dir.iterdir())[-1]
    print(f"\nBase scene saved: {base_path}")

    print(f"\nSTEP 2/2: Generating {args.num_variations} variations...")
    for i, variation_prompt in enumerate(variations):
        var_ts = datetime.now().strftime("%H%M%S")
        combined = f"{BASE_PROMPT}. Small objects: {variation_prompt}"
        run_scenesmith([
            f"+name=kitchen_var{i}_{var_ts}",
            f"experiment.prompts={json.dumps([combined])}",
            "floor_plan_agent.mode=room",
            "experiment.pipeline.start_stage=manipuland",
            f"experiment.pipeline.resume_from_path={base_path}",
        ], label=f"Variation {i+1}/{len(variations)}: {variation_prompt[:70]}...")

    print(f"\n{'='*60}\n  DONE! Base + {args.num_variations} variations\n  Base: {base_path}\n  All outputs: outputs/{today}/\n{'='*60}\n")

if __name__ == "__main__":
    main()
