"""Configuration for HSSD retrieval system."""

import logging
import os

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig

console_logger = logging.getLogger(__name__)


@dataclass
class HssdConfig:
    """Configuration for HSSD asset retrieval."""

    data_path: Path
    """Path to HSSD models directory (containing objects/ subdirectory).
    Ignored when azure_connection_string is set."""

    preprocessed_path: Path
    """Path to preprocessed data (indices, embeddings).
    When using Azure, preprocessed files are downloaded here automatically."""

    use_top_k: int = 5
    """Number of top CLIP candidates to consider before size ranking."""

    object_type_mapping: dict[str, str] | None = None
    """Map scenesmith ObjectType to HSSD categories."""

    azure_connection_string: str | None = None
    """Azure Blob Storage connection string. When set, meshes are streamed
    from blob instead of loaded from local data_path."""

    azure_container_name: str = "datasets"
    """Azure Blob Storage container name."""

    azure_blob_prefix: str = "hssd-models"
    """Blob path prefix for HSSD data (e.g., 'hssd-models')."""

    def __post_init__(self) -> None:
        """Validate configuration and set defaults."""
        self.data_path = Path(self.data_path)
        self.preprocessed_path = Path(self.preprocessed_path)

        # Check environment variable as fallback for azure connection string.
        if self.azure_connection_string is None:
            self.azure_connection_string = os.environ.get(
                "AZURE_HSSD_CONNECTION_STRING"
            )

        if self.use_azure:
            # Azure mode: preprocessed data will be downloaded on first use.
            # data_path is not needed for mesh loading (streamed from blob).
            console_logger.info("HSSD configured for Azure Blob Storage")
        else:
            # Local mode: validate paths exist.
            if not self.data_path.exists():
                raise FileNotFoundError(
                    f"HSSD data path does not exist: {self.data_path}"
                )
            if not self.preprocessed_path.exists():
                raise FileNotFoundError(
                    f"Preprocessed data path does not exist: {self.preprocessed_path}"
                )

        if self.object_type_mapping is None:
            self.object_type_mapping = {
                "FURNITURE": "large_objects",
                "MANIPULAND": "small_objects",
                "WALL_MOUNTED": "wall_objects",
                "CEILING_MOUNTED": "ceiling_objects",
            }

        console_logger.info(
            f"HSSD config initialized:\n"
            f"  data_path: {self.data_path}\n"
            f"  preprocessed_path: {self.preprocessed_path}\n"
            f"  azure: {self.use_azure}\n"
            f"  top_k: {self.use_top_k}"
        )

    @property
    def use_azure(self) -> bool:
        """Whether Azure Blob Storage is configured for mesh loading."""
        return self.azure_connection_string is not None

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "HssdConfig":
        """Create config from Hydra/OmegaConf nested structure.

        Args:
            cfg: HSSD config subtree (cfg.asset_manager.hssd).

        Returns:
            HssdConfig instance.
        """
        azure_connection_string = getattr(cfg, "azure_connection_string", None)
        azure_container_name = getattr(cfg, "azure_container_name", "datasets")
        azure_blob_prefix = getattr(cfg, "azure_blob_prefix", "hssd-models")

        return cls(
            data_path=Path(cfg.data_path),
            preprocessed_path=Path(cfg.preprocessed_path),
            use_top_k=cfg.use_top_k,
            object_type_mapping=dict(cfg.object_type_mapping),
            azure_connection_string=azure_connection_string,
            azure_container_name=azure_container_name,
            azure_blob_prefix=azure_blob_prefix,
        )
