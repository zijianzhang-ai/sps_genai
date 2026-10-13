import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from helper_lib.data_loader import get_assignment_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_model
from helper_lib.checkpoints import load_checkpoint
from helper_lib.evaluator import evaluate_model


def main():
    torch.manual_seed(42)
    device = "cpu"

    # Load CIFAR-10 and resize images to 64x64.
    full_train_loader = get_assignment_data_loader(
        "data/train", batch_size=64
    )
    test_loader = get_assignment_data_loader(
        "data/test", batch_size=64, train=False
    )

    # Use separate training, validation and test data.
    train_dataset, val_dataset = random_split(
        full_train_loader.dataset,
        [45000, 5000],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False
    )

    model = get_model("AssignmentCNN").to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device=device,
        epochs=10,
        checkpoint_dir="checkpoints_assignment2"
    )

    # Restore the model with the best validation accuracy.
    load_checkpoint(
        trained_model,
        optimizer,
        "checkpoints_assignment2/best_model.pth",
        device
    )

    avg_loss, accuracy = evaluate_model(
        trained_model,
        test_loader,
        criterion,
        device=device
    )

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()