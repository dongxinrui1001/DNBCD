from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    dataset_dir: Path = Path(r"F:\daimaFuxian\DNBCD\busi_project\dataset\Dataset_BUSI_with_GT")
    work_dir: Path = Path("runs/busi")
    model_name: str = "dnbcd"
    image_size: int = 128
    batch_size: int = 8
    epochs: int = 1
    seed: int = 42
    test_size: float = 0.20
    validation_size: float = 0.10

    @property
    def split_dir(self) -> Path:
        return self.work_dir / "split_data"

    @property
    def checkpoint_path(self) -> Path:
        return self.work_dir / f"{self.model_name}_best.keras"
