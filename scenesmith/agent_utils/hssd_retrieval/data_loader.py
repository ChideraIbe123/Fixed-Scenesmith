"""Data loading utilities for HSSD preprocessed indices and embeddings."""

from __future__ import annotations

import io
import json
import logging

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

if TYPE_CHECKING:
    from scenesmith.agent_utils.hssd_retrieval.config import HssdConfig

console_logger = logging.getLogger(__name__)


@dataclass
class HssdMeshMetadata:
    """Metadata for a single HSSD mesh."""

    mesh_id: str
    """HSSD mesh ID (SHA-1 hash)."""

    name: str
    """Human-readable object name."""

    up: str
    """Up vector as string "x,y,z"."""

    front: str
    """Front vector as string "x,y,z"."""

    wordnet_key: str
    """WordNet synset key this mesh belongs to."""


@dataclass
class HssdPreprocessedData:
    """Container for all preprocessed HSSD data."""

    metadata_by_wordnet: dict[str, list[HssdMeshMetadata]]
    """Maps WordNet synset keys to mesh metadata lists."""

    clip_embeddings: np.ndarray
    """CLIP embeddings array (N, 1024)."""

    embedding_index: list[str]
    """Maps array index to mesh ID."""

    object_categories: dict[str, list[str]]
    """Maps object types to WordNet synset keys."""

    _metadata_by_id: dict[str, HssdMeshMetadata] = field(
        init=False, default_factory=dict, repr=False
    )
    """Private O(1) lookup index from mesh_id to metadata."""

    def __post_init__(self):
        """Build metadata lookup index after initialization."""
        self._metadata_by_id = {
            m.mesh_id: m for meshes in self.metadata_by_wordnet.values() for m in meshes
        }

    def get_metadata(self, mesh_id: str) -> HssdMeshMetadata | None:
        """Get metadata for a specific mesh ID (O(1) lookup).

        Args:
            mesh_id: HSSD mesh ID to look up.

        Returns:
            Mesh metadata if found, None otherwise.
        """
        return self._metadata_by_id.get(mesh_id)

    def get_embedding_index(self, mesh_id: str) -> int | None:
        """Get the embedding array index for a mesh ID.

        Args:
            mesh_id: HSSD mesh ID to look up.

        Returns:
            Array index if found, None otherwise.
        """
        try:
            return self.embedding_index.index(mesh_id)
        except ValueError:
            return None


def load_preprocessed_data(preprocessed_path: Path) -> HssdPreprocessedData:
    """Load all preprocessed HSSD data.

    Args:
        preprocessed_path: Path to directory containing preprocessed files.

    Returns:
        Loaded preprocessed data.

    Raises:
        FileNotFoundError: If required files are missing.
        ValueError: If data format is invalid.
    """
    console_logger.info(f"Loading HSSD preprocessed data from {preprocessed_path}")

    index_path = preprocessed_path / "hssd_wnsynsetkey_index.json"
    embeddings_path = preprocessed_path / "clip_hssd_embeddings.npy"
    embedding_index_path = preprocessed_path / "clip_hssd_embeddings_index.yaml"
    categories_path = preprocessed_path / "object_categories.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not embedding_index_path.exists():
        raise FileNotFoundError(f"Embedding index not found: {embedding_index_path}")
    if not categories_path.exists():
        raise FileNotFoundError(f"Categories file not found: {categories_path}")

    with open(index_path, "r") as f:
        index_data = json.load(f)

    metadata_by_wordnet: dict[str, list[HssdMeshMetadata]] = {}
    total_entries = 0
    entries_with_orientation = 0
    entries_without_orientation = 0
    for wordnet_key, entries in index_data.items():
        metadata_list = []
        for entry in entries:
            total_entries += 1

            # Extract orientation fields (may be empty strings).
            up = entry.get("up", "")
            front = entry.get("front", "")

            # Track orientation availability for logging.
            if up and front:
                entries_with_orientation += 1
            else:
                entries_without_orientation += 1

            metadata = HssdMeshMetadata(
                mesh_id=entry["id"],
                name=entry["name"],
                up=up,
                front=front,
                wordnet_key=wordnet_key,
            )
            metadata_list.append(metadata)
        metadata_by_wordnet[wordnet_key] = metadata_list

    console_logger.info(
        f"Loaded {total_entries} HSSD entries: {entries_with_orientation} with "
        f"orientation data ({entries_with_orientation/total_entries*100:.1f}%), "
        f"{entries_without_orientation} without "
        f"({entries_without_orientation/total_entries*100:.1f}%)"
    )

    clip_embeddings = np.load(embeddings_path)
    console_logger.info(
        f"Loaded CLIP embeddings: shape={clip_embeddings.shape}, "
        f"dtype={clip_embeddings.dtype}"
    )

    with open(embedding_index_path, "r") as f:
        embedding_index = yaml.safe_load(f)

    with open(categories_path, "r") as f:
        categories_data = json.load(f)

    object_categories = {
        category: wordnet_keys
        for category, wordnet_keys in categories_data.items()
        if category != "available_categories"
    }

    console_logger.info(
        f"Loaded {len(metadata_by_wordnet)} WordNet categories, "
        f"{len(embedding_index)} mesh embeddings, "
        f"{len(object_categories)} object categories"
    )

    return HssdPreprocessedData(
        metadata_by_wordnet=metadata_by_wordnet,
        clip_embeddings=clip_embeddings,
        embedding_index=embedding_index,
        object_categories=object_categories,
    )


def construct_hssd_mesh_path(hssd_dir_path: Path, mesh_id: str) -> Path:
    """Construct the file path for an HSSD mesh.

    Args:
        hssd_dir_path: Root directory of HSSD models (containing objects/).
        mesh_id: HSSD mesh ID (SHA-1 hash).

    Returns:
        Path to the GLB mesh file.

    Raises:
        FileNotFoundError: If the constructed path does not exist.
    """
    first_char = mesh_id[0]
    mesh_path = hssd_dir_path / "objects" / first_char / f"{mesh_id}.glb"

    if not mesh_path.exists():
        raise FileNotFoundError(f"HSSD mesh not found: {mesh_path}")

    return mesh_path


def download_hssd_mesh_from_azure(config: HssdConfig, mesh_id: str) -> io.BytesIO:
    """Download an HSSD mesh from Azure Blob Storage into memory.

    Args:
        config: HSSD config with Azure connection details.
        mesh_id: HSSD mesh ID (SHA-1 hash).

    Returns:
        BytesIO buffer containing the GLB file data.

    Raises:
        FileNotFoundError: If the blob does not exist.
        RuntimeError: If Azure download fails.
    """
    from azure.core.exceptions import ResourceNotFoundError
    from azure.storage.blob import BlobClient

    first_char = mesh_id[0]
    blob_name = f"{config.azure_blob_prefix}/objects/{first_char}/{mesh_id}.glb"

    try:
        blob_client = BlobClient.from_connection_string(
            config.azure_connection_string,
            container_name=config.azure_container_name,
            blob_name=blob_name,
        )
        stream = blob_client.download_blob()
        buffer = io.BytesIO(stream.readall())
        buffer.seek(0)
        return buffer
    except ResourceNotFoundError:
        raise FileNotFoundError(
            f"HSSD mesh not found in Azure blob: {blob_name}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download mesh {mesh_id} from Azure: {e}")


def ensure_preprocessed_from_azure(config: HssdConfig) -> Path:
    """Download preprocessed HSSD data from Azure Blob if not cached locally.

    Downloads the 4 preprocessed files (index, embeddings, embedding index,
    categories) to config.preprocessed_path. Skips files that already exist.

    Args:
        config: HSSD config with Azure connection details.

    Returns:
        Path to local directory containing preprocessed files.
    """
    from azure.storage.blob import ContainerClient

    preprocessed_path = config.preprocessed_path
    preprocessed_path.mkdir(parents=True, exist_ok=True)

    files = [
        "hssd_wnsynsetkey_index.json",
        "clip_hssd_embeddings.npy",
        "clip_hssd_embeddings_index.yaml",
        "object_categories.json",
    ]

    container_client = ContainerClient.from_connection_string(
        config.azure_connection_string,
        container_name=config.azure_container_name,
    )

    for filename in files:
        local_path = preprocessed_path / filename
        if local_path.exists():
            console_logger.debug(f"Preprocessed file already cached: {filename}")
            continue

        blob_name = f"{config.azure_blob_prefix}/preprocessed/{filename}"
        console_logger.info(f"Downloading preprocessed file from Azure: {blob_name}")

        blob_client = container_client.get_blob_client(blob_name)
        data = blob_client.download_blob().readall()

        with open(local_path, "wb") as f:
            f.write(data)

        console_logger.info(
            f"Downloaded {filename} ({len(data) / 1024 / 1024:.1f} MB)"
        )

    return preprocessed_path
