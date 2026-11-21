from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from helper_lib.data_loader import get_assignment_data_loader
from helper_lib.model import UNet, DiffusionModel, offset_cosine_diffusion_schedule


torch.manual_seed(42)

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

print(f"Using device: {device}")

project_dir = Path(__file__).resolve().parent

# Batch size chosen for this implementation.
BATCH_SIZE = 32

# Load the complete CIFAR-10 training and test sets.
train_loader = get_assignment_data_loader(
    str(project_dir / "data" / "train"),
    batch_size=BATCH_SIZE,
    train=True
)

test_loader = get_assignment_data_loader(
    str(project_dir / "data" / "test"),
    batch_size=BATCH_SIZE,
    train=False
)

print(f"Training images: {len(train_loader.dataset)}")
print(f"Test images: {len(test_loader.dataset)}")

# Use the professor's UNet and diffusion schedule.
unet = UNet(
    image_size=64,
    num_channels=3,
    embedding_dim=32
)

diffusion_model = DiffusionModel(
    unet,
    offset_cosine_diffusion_schedule
).to(device)

# Check one small batch without updating model weights.
images, _ = next(iter(train_loader))
sample_images = images[:2].to(device)

noise_variances = torch.full(
    (sample_images.size(0), 1, 1, 1),
    0.5,
    device=device
)

diffusion_model.network.eval()

with torch.no_grad():
    predicted_noise = diffusion_model.network(
        sample_images,
        noise_variances
    )

print(f"Input shape: {sample_images.shape}")
print(f"Output shape: {predicted_noise.shape}")
print("Diffusion data and model check passed.")

# Calculate per-channel normalization statistics.
NUM_CHANNELS = 3

mean = torch.zeros(NUM_CHANNELS)
std = torch.zeros(NUM_CHANNELS)
total_samples = 0

train_loader_for_stats = DataLoader(
    train_loader.dataset,
    batch_size=BATCH_SIZE
)

for imgs, _ in train_loader_for_stats:
    batch_size = imgs.size(0)

    imgs_flat = imgs.view(
        batch_size,
        NUM_CHANNELS,
        -1
    )

    batch_mean = imgs_flat.mean(dim=(0, 2))
    batch_std = imgs_flat.std(dim=(0, 2))

    mean += batch_mean * batch_size
    std += batch_std * batch_size
    total_samples += batch_size

mean /= total_samples
std /= total_samples

print("Normalization stats - Mean:", mean, "Std:", std)

mean = mean.reshape(1, NUM_CHANNELS, 1, 1).to(device)
std = std.reshape(1, NUM_CHANNELS, 1, 1).to(device)

diffusion_model.set_normalizer(mean, std)

print("Normalization setup complete.")

optimizer = torch.optim.AdamW(
    diffusion_model.network.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

loss_fn = nn.L1Loss()
EPOCHS = 13

checkpoint_dir = project_dir / "checkpoints_assignment4_diffusion"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

latest_path = checkpoint_dir / "diffusion_latest.pth"
best_path = checkpoint_dir / "diffusion_best.pth"

start_epoch = 0
best_val_loss = float("inf")

# Resume from the last completed epoch if a checkpoint exists.
if latest_path.exists():
    saved = torch.load(latest_path, map_location=device)

    diffusion_model.network.load_state_dict(
        saved["model_state_dict"]
    )
    diffusion_model.ema_network.load_state_dict(
        saved["ema_model_state_dict"]
    )
    optimizer.load_state_dict(saved["optimizer_state_dict"])

    diffusion_model.set_normalizer(
        saved["normalizer_mean"],
        saved["normalizer_std"]
    )

    start_epoch = saved["epoch"]
    best_val_loss = saved["best_val_loss"]

    print(f"Resuming after epoch {start_epoch}")


for epoch in range(start_epoch, EPOCHS):
    diffusion_model.train()
    training_losses = []

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]",
        ncols=120
    )

    for images, _ in progress:
        images = images.to(device)

        loss = diffusion_model.train_step(
            images,
            optimizer,
            loss_fn
        )

        training_losses.append(loss)
        progress.set_postfix(loss=f"{loss:.4f}")

    average_train_loss = sum(training_losses) / len(training_losses)

    diffusion_model.eval()
    validation_losses = []

    with torch.no_grad():
        for images, _ in tqdm(
            test_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]",
            ncols=120
        ):
            loss = diffusion_model.test_step(
                images.to(device),
                loss_fn
            )
            validation_losses.append(loss)

    average_val_loss = sum(validation_losses) / len(validation_losses)

    improved = average_val_loss < best_val_loss
    best_val_loss = min(best_val_loss, average_val_loss)

    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": diffusion_model.network.state_dict(),
        "ema_model_state_dict": diffusion_model.ema_network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "normalizer_mean": diffusion_model.normalizer_mean,
        "normalizer_std": diffusion_model.normalizer_std,
        "train_loss": average_train_loss,
        "val_loss": average_val_loss,
        "best_val_loss": best_val_loss
    }

    torch.save(checkpoint, latest_path)

    if improved:
        torch.save(checkpoint, best_path)

    print(
        f"\nEpoch {epoch + 1}: "
        f"Train loss = {average_train_loss:.4f}, "
        f"Val loss = {average_val_loss:.4f}"
    )
    print(f"Latest checkpoint saved to: {latest_path}")

print("\nDiffusion training finished.")