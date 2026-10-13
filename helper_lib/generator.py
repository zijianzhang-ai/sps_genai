import torch
import matplotlib.pyplot as plt
import math


def generate_samples(model, device, num_samples=10):
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
        rows, cols,
        figsize=(10, 2 * rows),
        squeeze=False
    )

    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            ax.imshow(samples[i].squeeze(), cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.show()