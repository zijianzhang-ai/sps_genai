from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from helper_lib.model import get_model


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint_path = (
    Path(__file__).resolve().parent
    / "checkpoints_assignment3"
    / "assignment3_gan.pth"
)

generator, _ = get_model("Assignment3GAN")

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

generator.load_state_dict(
    checkpoint["generator_state_dict"]
)

generator = generator.to(device)
generator.eval()

with torch.no_grad():
    noise = torch.randn(
        25,
        100,
        device=device
    )

    generated_images = generator(noise).cpu()

grid = make_grid(
    generated_images,
    nrow=5,
    normalize=True,
    value_range=(-1, 1)
)

output_path = (
    Path(__file__).resolve().parent
    / "assignment3_generated_images.png"
)

plt.figure(figsize=(6, 6))
plt.imshow(
    grid.permute(1, 2, 0).squeeze(),
    cmap="gray"
)
plt.axis("off")
plt.tight_layout()
plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)
plt.show()

print(generated_images.shape)
print(output_path)