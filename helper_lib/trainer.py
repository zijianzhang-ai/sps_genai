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

def train_gan(
    model,
    data_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10
):
    gen, critic = model
    opt_gen, opt_critic = optimizer

    gen = gen.to(device)
    critic = critic.to(device)

    gen.train()
    critic.train()

    z_dim = 100
    n_critic = 5
    clip_value = 0.01

    for epoch in range(epochs):
        train_loader_with_progress = tqdm(
            iterable=data_loader,
            ncols=120,
            desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for batch_number, (real, _) in enumerate(
            train_loader_with_progress
        ):
            real = real.to(device)
            batch_size = real.size(0)

            # === Train Critic ===
            for _ in range(n_critic):
                noise = torch.randn(
                    batch_size, z_dim, 1, 1
                ).to(device)

                fake = gen(noise).detach()

                critic_real = critic(real).mean()
                critic_fake = critic(fake).mean()

                loss_critic = -(critic_real - critic_fake)

                critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

                # Weight clipping
                for p in critic.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            # === Train Generator ===
            noise = torch.randn(
                batch_size, z_dim, 1, 1
            ).to(device)

            fake = gen(noise)
            loss_gen = -critic(fake).mean()

            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_number % 100 == 0:
                train_loader_with_progress.set_postfix(
                    {
                        "Batch": (
                            f"{batch_number}/{len(data_loader)}"
                        ),
                        "D loss": f"{loss_critic.item():.4f}",
                        "G loss": f"{loss_gen.item():.4f}",
                    }
                )

    return model

def train_assignment3_gan(
    model,
    data_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10
):
    generator, discriminator = model
    optimizer_generator, optimizer_discriminator = optimizer

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    generator.train()
    discriminator.train()

    z_dim = 100

    for epoch in range(epochs):
        progress = tqdm(
            iterable=data_loader,
            ncols=120,
            desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for real_images, _ in progress:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_labels = torch.ones(
                batch_size,
                1,
                device=device
            )

            fake_labels = torch.zeros(
                batch_size,
                1,
                device=device
            )

            # Train Discriminator
            optimizer_discriminator.zero_grad()

            real_predictions = discriminator(
                real_images
            )

            real_loss = criterion(
                real_predictions,
                real_labels
            )

            noise = torch.randn(
                batch_size,
                z_dim,
                device=device
            )

            fake_images = generator(noise)

            fake_predictions = discriminator(
                fake_images.detach()
            )

            fake_loss = criterion(
                fake_predictions,
                fake_labels
            )

            discriminator_loss = (
                real_loss + fake_loss
            ) / 2

            discriminator_loss.backward()
            optimizer_discriminator.step()

            # Train Generator
            optimizer_generator.zero_grad()

            generated_predictions = discriminator(
                fake_images
            )

            generator_loss = criterion(
                generated_predictions,
                real_labels
            )

            generator_loss.backward()
            optimizer_generator.step()

            progress.set_postfix(
                {
                    "D loss": (
                        f"{discriminator_loss.item():.4f}"
                    ),
                    "G loss": (
                        f"{generator_loss.item():.4f}"
                    )
                }
            )

    return model
    
def train_diffusion(
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
        train_loader_with_progress = tqdm(
            iterable=data_loader,
            ncols=120,
            desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for images, _ in train_loader_with_progress:
            images = images.to(device)

            loss = model.train_step(
                images,
                optimizer,
                criterion
            )

            train_loader_with_progress.set_postfix(
                loss=f"{loss:.4f}"
            )

    return model   