from pathlib import Path

import torch
import torch.nn as nn

from helper_lib.data_loader import (
    get_assignment3_data_loader
)
from helper_lib.model import get_model
from helper_lib.trainer import (
    train_assignment3_gan
)


torch.manual_seed(42)

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

print(f"Using device: {device}")

data_loader = get_assignment3_data_loader(
    data_dir="data",
    batch_size=64,
    train=True
)

model = get_model("Assignment3GAN")
generator, discriminator = model

criterion = nn.BCELoss()

optimizer_generator = torch.optim.Adam(
    generator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

optimizer_discriminator = torch.optim.Adam(
    discriminator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

EPOCHS = 20

trained_model = train_assignment3_gan(
    model=model,
    data_loader=data_loader,
    criterion=criterion,
    optimizer=(
        optimizer_generator,
        optimizer_discriminator
    ),
    device=device,
    epochs=EPOCHS
)

generator, discriminator = trained_model

checkpoint_dir = (
    Path(__file__).resolve().parent
    / "checkpoints_assignment3"
)

checkpoint_dir.mkdir(
    parents=True,
    exist_ok=True
)

checkpoint_path = (
    checkpoint_dir
    / "assignment3_gan.pth"
)

torch.save(
    {
        "generator_state_dict": (
            generator.state_dict()
        ),
        "discriminator_state_dict": (
            discriminator.state_dict()
        ),
        "optimizer_generator_state_dict": (
            optimizer_generator.state_dict()
        ),
        "optimizer_discriminator_state_dict": (
            optimizer_discriminator.state_dict()
        ),
        "epochs": EPOCHS
    },
    checkpoint_path
)

print(f"Model saved to: {checkpoint_path}")