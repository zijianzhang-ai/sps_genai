import os
import torch
from tqdm import tqdm
from .checkpoints import save_checkpoint

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10,
    checkpoint_dir="checkpoints"
):
    model = model.to(device)
    best_accuracy = -1.0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for inputs, labels in progress:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            predicted = outputs.argmax(dim=1)
            train_loss += loss.item() * labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_accuracy = 100 * train_correct / train_total

        # Evaluate without updating model weights
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                predicted = outputs.argmax(dim=1)

                val_loss += loss.item() * labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_accuracy = 100 * val_correct / val_total

        print(
            f"Train loss: {train_loss:.4f}, "
            f"Train accuracy: {train_accuracy:.2f}% | "
            f"Val loss: {val_loss:.4f}, "
            f"Val accuracy: {val_accuracy:.2f}%"
        )

        # Save a checkpoint after every epoch
        checkpoint_path = save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            val_loss,
            val_accuracy,
            checkpoint_dir
        )

        # Save the checkpoint with the highest validation accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device
            )
            torch.save(
                checkpoint,
                os.path.join(checkpoint_dir, "best_model.pth")
            )

    return model

def train_vae_model(
    model,
    data_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10
):
    model = model.to(device)
    model.train()

    for epoch in range(epochs):
        progress = tqdm(
            data_loader,
            desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for inputs, _ in progress:
            inputs = inputs.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(inputs)
            loss = criterion(recon, inputs, mu, logvar)
            loss.backward()
            optimizer.step()

            progress.set_postfix(loss=f"{loss.item():.4f}")

    print("Finished Training")
    return model


def vae_loss_function(recon_x, x, mu, logvar):
    # Same loss formula as the professor's VAE notebook.
    beta = 500

    BCE = torch.nn.functional.binary_cross_entropy(
        recon_x, x, reduction="sum"
    )

    KLD = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return beta * BCE + KLD