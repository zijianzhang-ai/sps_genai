import torch
import matplotlib.pyplot as plt
import math
from torchvision.utils import make_grid
from .model import DiffusionModel

def generate_samples(model, device, num_samples=10, diffusion_steps=100):
    # GAN generation
    if isinstance(model, tuple):
        gen, _ = model

        gen = gen.to(device)
        gen.eval()

        z_dim = 100

        with torch.no_grad():
            noise = torch.randn(
                num_samples, z_dim, 1, 1
            ).to(device)

            fake = gen(noise).detach().cpu()
            grid = make_grid(fake, normalize=True)

        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")
        plt.show()
        return
    
    # Diffusion generation
    if isinstance(model, DiffusionModel):
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            samples = model.generate(
                num_images=num_samples,
                diffusion_steps=diffusion_steps,
                image_size=64
            ).cpu()

            grid = make_grid(
                samples,
                normalize=True
            )

        plt.imshow(
            grid.permute(1, 2, 0)
        )
        plt.axis("off")
        plt.show()
        return

    # VAE generation
    model = model.to(device)
    model.eval()

    # Sample points from the standard normal latent distribution.
    with torch.no_grad():
        z = torch.randn(num_samples, 2).to(device)
        samples = model.decoder(z).cpu().numpy()

    # Arrange the generated images in a grid.
    cols = 5
    rows = math.ceil(num_samples / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(10, 2 * rows),
        squeeze=False
    )

    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            ax.imshow(samples[i].squeeze(), cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.show()