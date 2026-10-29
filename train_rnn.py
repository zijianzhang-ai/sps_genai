import re
import requests
import numpy as np
import torch
import torch.nn as nn

from collections import Counter
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.rnn_model import LSTMModel, TextDataset


torch.manual_seed(42)
np.random.seed(42)

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

print(f"Using device: {device}")

# Load and preprocess Count of Monte Cristo
url = "https://www.gutenberg.org/cache/epub/1184/pg1184.txt"

text = requests.get(url).text

# Keep only the main body
start_idx = text.find("Chapter 1")
end_idx = text.rfind("Chapter 5")
text = text[start_idx:end_idx]

# Pre-processing
text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
text = text.lower()

# Tokenization
tokens = text.split()

# Vocabulary construction
counter = Counter(tokens)

vocab = {
    word: idx + 2
    for idx, (word, _) in enumerate(
        counter.most_common(9998)
    )
}

vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

inv_vocab = {
    idx: word
    for word, idx in vocab.items()
}

# Encode tokens
encoded = [
    vocab.get(word, vocab["<UNK>"])
    for word in tokens
]

print(f"Number of tokens: {len(tokens)}")
print(f"Vocabulary size: {len(vocab)}")

train_datasets = TextDataset(encoded)

train_loader = DataLoader(
    train_datasets,
    batch_size=64,
    shuffle=True
)

model = LSTMModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Local CPU configuration.
# The professor's Practical uses 15 epochs.
EPOCHS = 1

for epoch in range(EPOCHS):
    total_loss = 0

    train_loader_with_progress = tqdm(
        iterable=train_loader,
        ncols=120,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for batch_number, (inputs, targets) in enumerate(
        train_loader_with_progress
    ):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs, _ = model(inputs)

        loss = criterion(
            outputs.view(-1, outputs.size(-1)),
            targets.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (
            batch_number % 100 == 0
            or batch_number == len(train_loader) - 1
        ):
            train_loader_with_progress.set_postfix(
                {
                    "avg loss": (
                        f"{total_loss / (batch_number + 1):.4f}"
                    )
                }
            )

checkpoint_dir = (
    Path(__file__).resolve().parent
    / "checkpoints_rnn"
)

checkpoint_dir.mkdir(
    parents=True,
    exist_ok=True
)

checkpoint_path = (
    checkpoint_dir
    / "rnn_model.pth"
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "inv_vocab": inv_vocab,
    },
    checkpoint_path
)

print(f"Model saved to: {checkpoint_path}")