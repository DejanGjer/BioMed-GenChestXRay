# Generative Adversarial Networks for Chest X-ray generation

## Presentation

Presentation slides (serbian) can be found [here](https://docs.google.com/presentation/d/1gfiVajAiR8EHV__tFSZVNDnqCnU982NUOmHJSt6ENSw/edit?usp=sharing).


## Training setup

* Install the repository:

```sh
pip install -e .
```

* Log in to wandb for logging training:

```sh
wandb login #and enter API key when prompted
```

* Download the dataset from [here](https://www.kaggle.com/datasets/nih-chest-xrays/data?select=images_001), and extract the zip to get the `images` folder

Note: we're only using the first 5000 images from the dataset

* Set up the training configuration at `training_config.yml`, by changing the `training_dataset_location` and other training hyperparameters

* Run training:

```sh
python src/training.py
```

* Run inference:

```sh
python src/inference.py --ckpt_dir /home/mmilenkovic/git/BioMed-GenChestXRay/models/202407130110/ --num_images 5
```