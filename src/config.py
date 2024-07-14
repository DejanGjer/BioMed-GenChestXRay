from typing import Optional
from dataclasses import dataclass
import yaml

@dataclass
class TrainingConfig:
    model_name: str
    ngf: int
    dataset_location: str
    training_images_to_use: int
    log_interval: int
    batch_size: int
    max_summary_images: int
    image_size: int
    channels: int
    z_size: int
    lr_critic: float
    lr_generator: float
    epochs: int
    num_workers: int
    clip_value: float
    wandb_relogin: bool
    wandb_api_key: str
    c_times: Optional[int]=1
    g_times: Optional[int]=1

    def __post_init__(self):
        assert (
            self.max_summary_images <= self.batch_size
        ), "We can only write to Tensorboard as many images as there are in a batch"
        assert not (self.c_times!=1 and self.g_times!=1),"Can't train both critic and generator more than the other"

    @classmethod
    def from_yaml(cls, yaml_file: str) -> "TrainingConfig":
        with open(yaml_file, "r") as f:
            yaml_data = yaml.safe_load(f)
        return cls(**yaml_data)
