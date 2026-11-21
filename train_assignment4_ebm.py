from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from helper_lib.energy_model import EnergyModel, generate_samples, Buffer, EBM
from tqdm import tqdm

torch.manual_seed(42)
device = torch.device("cpu")
project_dir = Path(__file__).resolve().parent

# EBM uses images scaled to [-1, 1].
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5),
    ),
])

train_dataset = datasets.CIFAR10(
    root=str(project_dir / "data" / "train"),
    train=True,
    download=True,
    transform=transform,
)

BATCH_SIZE = 128

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

energy_model = EnergyModel().to(device)

images, labels = next(iter(train_loader))
images = images[:2].to(device)

energy_model.eval()
with torch.no_grad():
    energies = energy_model(images)

print(f"Using device: {device}")
print(f"Training images: {len(train_dataset)}")
print(f"Input shape: {images.shape}")
print(f"Energy output shape: {energies.shape}")
print("EBM data and model check passed.")

# Test sampling with two random RGB images.
initial_images = torch.rand(
    2, 3, 32, 32, device=device
) * 2 - 1

samples = generate_samples(
    nn_energy_model=energy_model,
    inp_imgs=initial_images,
    steps=10,
    step_size=10.0,
    noise_std=0.005,
)

assert samples.shape == initial_images.shape
assert torch.isfinite(samples).all()
assert not samples.requires_grad
assert samples.min() >= -1.0 and samples.max() <= 1.0

print(f"Sample shape: {samples.shape}")
print(
    f"Sample range: "
    f"{samples.min().item():.4f} to {samples.max().item():.4f}"
)
print("EBM sampling check passed.")

# Use only two images for this quick buffer check.
sample_buffer = Buffer(
    model=energy_model,
    device=device,
    batch_size=2,
)

buffer_samples = sample_buffer.sample_new_exmps(
    steps=10,
    step_size=10.0,
    noise=0.005,
)

assert buffer_samples.shape == (2, 3, 32, 32)
assert torch.isfinite(buffer_samples).all()
assert len(sample_buffer.examples) == 4

print(f"Buffer sample shape: {buffer_samples.shape}")
print(f"Images stored in buffer: {len(sample_buffer.examples)}")
print("EBM buffer check passed.")

# Start with one complete training epoch.
EPOCHS = 10

ebm = EBM(
    model=energy_model,
    alpha=0.1,
    steps=60,
    step_size=10,
    noise=0.005,
    device=device,
    batch_size=BATCH_SIZE,
)

optimizer = torch.optim.Adam(
    energy_model.parameters(),
    lr=0.0001,
    betas=(0.0, 0.999),
)

checkpoint_dir = project_dir / "checkpoints_assignment4_ebm"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = checkpoint_dir / "ebm_latest.pth"

start_epoch = 0

# Resume model, optimizer, and sampling buffer.
if checkpoint_path.exists():
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    energy_model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    ebm.buffer.examples = [
        image.to(device)
        for image in checkpoint["buffer_examples"]
    ]

    start_epoch = checkpoint["epoch"]
    print(f"Resuming after epoch {start_epoch}")

for epoch in range(start_epoch, EPOCHS):
    totals = {
        "loss": 0.0,
        "reg": 0.0,
        "cdiv": 0.0,
        "real": 0.0,
        "fake": 0.0,
    }
    batches = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} [EBM]",
        ncols=120,
    )

    for real_images, _ in progress:
        metrics = ebm.train_step(
            real_images.to(device),
            optimizer,
        )

        for name, value in metrics.items():
            totals[name] += value

        batches += 1

        progress.set_postfix(
            loss=f"{metrics['loss']:.4f}",
            real=f"{metrics['real']:.4f}",
            fake=f"{metrics['fake']:.4f}",
        )

    averages = {
        name: total / batches
        for name, total in totals.items()
    }

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": energy_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "buffer_examples": [
                image.detach().cpu().clone()
                for image in ebm.buffer.examples
            ],
            "metrics": averages,
        },
        checkpoint_path,
    )

    print(
        f"Epoch {epoch + 1}: "
        + ", ".join(
            f"{name}={value:.4f}"
            for name, value in averages.items()
        )
    )
    print(f"Checkpoint saved to: {checkpoint_path}")

print("EBM training finished.")