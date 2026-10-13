from helper_lib.data_loader import get_fashion_data_loader
from helper_lib.trainer import train_vae_model, vae_loss_function
from helper_lib.model import get_model
from helper_lib.generator import generate_samples
import torch.optim as optim

device = "cpu"

train_loader = get_fashion_data_loader(
    "data/train", batch_size=64
)
test_loader = get_fashion_data_loader(
    "data/test", batch_size=64, train=False
)

vae = get_model("VAE")
criterion = vae_loss_function
optimizer = optim.Adam(vae.parameters(), lr=0.001)

train_vae_model(
    vae, train_loader, criterion, optimizer,
    device=device, epochs=5
)

generate_samples(vae, device, num_samples=10)