from typing import Any
import os
import yaml
import torch
from datetime import datetime

# from torch.utils.tensorboard import SummaryWriter
import wandb
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
from tqdm import tqdm

from critics import FCCritic, DCGANCritic
from generators import FCGenerator, DCGANGenerator
from dataset import XRayDataset
from config import TrainingConfig

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Boilerplate code for using CUDA for faster training
CUDA = torch.cuda.is_available()  # Use CUDA for faster training
OVERRIDE_CKPT=True

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def save_model(model:torch.nn.Module,config:Any):
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d%H%M")
    models_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"models")
    if OVERRIDE_CKPT:
        os.rmdir(models_dir)
    save_dir=os.path.join(models_dir,timestamp)
    os.makedirs(save_dir,exist_ok=True)
    save_path=os.path.join(save_dir,"x_ray_generator.pt")
    torch.save(model,save_path)
    config_path = os.path.join(save_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config.__dict__, f)
    print("Saved model to ",save_dir)

def train(training_config: TrainingConfig):
    # Initialize the Tensorboard summary. Logs will end up in runs directory
    # summary_writer = SummaryWriter()

    # Initialize the critic and the generator.
    critic, generator = None, None
    if training_config.model_name == "FC":
        critic=FCCritic.from_config(training_config)
        # critic = FCCritic(training_config.image_size, training_config.channels)
        generator=FCGenerator.from_config(training_config)
        # generator = FCGenerator(
        #     training_config.image_size, training_config.channels, training_config.z_size
        # )
    elif training_config.model_name == "DC":
        critic = DCGANCritic.from_config(training_config)
        # critic = DCGANCritic(training_config.image_size, training_config.channels, training_config.ngf)
        generator=DCGANGenerator.from_config(training_config)
        # generator = DCGANGenerator(
        #     training_config.image_size, training_config.channels, training_config.z_size, training_config.ngf
        # )
        critic.apply(weights_init)
        generator.apply(weights_init)
    else:
        raise ValueError(f"Unknown model name: {training_config.model_name}")

    critic.to(DEVICE)
    generator.to(DEVICE)

    # Initialize the data set. NOTE: You can pass total_images argument to avoid loading the whole dataset.
    data_set = XRayDataset(
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

    optimizer_c = torch.optim.RMSprop(critic.parameters(), lr=training_config.lr_critic)
    optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=training_config.lr_generator)

    critic_lr_lambda = lambda epoch: max(0, 1 - epoch / training_config.epochs)
    generator_lr_lambda = lambda epoch: max(0, 1 - epoch / training_config.epochs)

    # Create schedulers for critic and generator
    scheduler_critic = LambdaLR(optimizer_c, lr_lambda=critic_lr_lambda)
    scheduler_generator = LambdaLR(optimizer_g, lr_lambda=generator_lr_lambda)

    #TODO: add option to train generator more than critic
    assert training_config.g_times==1
    for epoch in range(training_config.epochs):
        c_times = training_config.c_times

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
            scheduler_critic.step(epoch=epoch)

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
                scheduler_generator.step(epoch=epoch)

                # summary_writer.add_images("Generated images", gen_imgs[:MAX_SUMMARY_IMAGES], global_step)
                images = wandb.Image(
                    gen_imgs[: training_config.max_summary_images],
                    caption="Top: Output, Bottom: Input",
                )
                wandb.log({"Generated images": images}, step=global_step)
    save_model(model=generator,config=training_config)
    

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
