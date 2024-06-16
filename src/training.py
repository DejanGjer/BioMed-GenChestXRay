import os
from dataclasses import dataclass
import yaml
import torch
import torchvision

# from torch.utils.tensorboard import SummaryWriter
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from critics import FCCritic
from generators import FCGenerator
from dataset import FacesDataSet

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Boilerplate code for using CUDA for faster training
CUDA = torch.cuda.is_available()  # Use CUDA for faster training


# TODO: use all values here
@dataclass
class TrainingConfig:
    dataset_location: str
    training_images_to_use: int
    batch_size: int
    max_summary_images: int
    image_size: int
    channels: int
    z_size: int
    lr: float
    epochs: int
    num_workers: int
    clip_value: float
    wandb_relogin: bool
    wandb_api_key: str

    def __post_init__(self):
        assert (
            self.max_summary_images <= self.batch_size
        ), "We can only write to Tensorboard as many images as there are in a batch"

    @classmethod
    def from_yaml(cls, yaml_file: str) -> "TrainingConfig":
        with open(yaml_file, "r") as f:
            yaml_data = yaml.safe_load(f)
        return cls(**yaml_data)


def train(training_config: TrainingConfig):
    # Initialize the Tensorboard summary. Logs will end up in runs directory
    # summary_writer = SummaryWriter()

    # Initialize the critic and the generator. NOTE: You can use other classes from critis.py and generators.py here.
    critic = FCCritic(training_config.image_size, training_config.channels)
    generator = FCGenerator(
        training_config.image_size, training_config.channels, training_config.z_size
    )

    critic.to(DEVICE)
    generator.to(DEVICE)

    # Initialize the data set. NOTE: You can pass total_images argument to avoid loading the whole dataset.
    data_set = FacesDataSet(
        img_size=training_config.image_size,
        crop_size=training_config.image_size,
        total_images=training_config.training_images_to_use,
        image_dir=training_config.dataset_location,
    )
    total_iterations = len(data_set) // training_config.batch_size
    data_loader = DataLoader(
        data_set,
        batch_size=training_config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=training_config.num_workers,
        drop_last=True,
    )

    optimizer_c = torch.optim.RMSprop(critic.parameters(), lr=training_config.lr)
    optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=training_config.lr)

    for epoch in range(training_config.epochs):
        if epoch < 10:
            c_times = 100
        else:
            c_times = 5

        for i, real_img_batch in tqdm(
            enumerate(data_loader),
            total=total_iterations,
            desc=f"Epoch: {epoch}",
            unit="batches",
        ):
            # Global step is used in tensorboard so that we have unique number for each batch of data
            global_step = epoch * total_iterations + i
            z_batch = torch.randn((training_config.batch_size, training_config.z_size))
            # model = FCGenerator(IMG_SIZE, CHANNELS, Z_SIZE)

            # This is just boilerplate if you're using CUDA - All inputs to the network need to be on the same device
            real_img_batch = real_img_batch.to(DEVICE)
            z_batch = z_batch.to(DEVICE)

            fake_img_batch = generator(z_batch).detach()

            optimizer_c.zero_grad()

            out_real = critic(real_img_batch)
            out_fake = critic(fake_img_batch)
            loss_c = -(torch.mean(out_real) - torch.mean(out_fake))

            # summary_writer.add_scalar("Critic loss", loss_c, global_step)
            wandb.log({"Critic loss": loss_c}, step=global_step)
            # Calculate the gradients with respect to the input
            loss_c.backward()
            # Apply backward prop
            optimizer_c.step()
            # Clip the weights of the critic to satisfy the Lipschitz constraint
            for p in critic.parameters():
                p.data.clamp_(-training_config.clip_value, training_config.clip_value)

            # Train the generator only after the critic has been trained c_times
            if i % c_times == 0:
                optimizer_g.zero_grad()
                gen_imgs = generator(z_batch)
                out_fake = critic(gen_imgs)
                # Adversarial loss
                loss_g = -torch.mean(out_fake)
                # summary_writer.add_scalar("Generator loss", loss_g, global_step)
                wandb.log({"Generator loss": loss_g}, step=global_step)

                # Calculate the gradients with respect to the input
                loss_g.backward()
                # Apply backward prop
                optimizer_g.step()

                # summary_writer.add_images("Generated images", gen_imgs[:MAX_SUMMARY_IMAGES], global_step)
                images = wandb.Image(
                    gen_imgs[: training_config.max_summary_images],
                    caption="Top: Output, Bottom: Input",
                )
                wandb.log({"Generated images": images}, step=global_step)


if __name__ == "__main__":
    # login to wandb
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(os.path.dirname(script_dir), "training_config.yml")
    training_config = TrainingConfig.from_yaml(config_file)
    if training_config.wandb_relogin:
        wandb.login(key=training_config.wandb_api_key)
    run = wandb.init(project="DCGAN Demo")

    train(training_config=training_config)

    wandb.finish()
