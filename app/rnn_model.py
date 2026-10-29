import torch
import torch.nn as nn
from torch.utils.data import Dataset

SEQ_LEN = 30


class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data) - SEQ_LEN

    def __getitem__(self, idx):
        return (
            torch.tensor(
                self.data[idx:idx + SEQ_LEN]
            ),
            torch.tensor(
                self.data[idx + 1:idx + SEQ_LEN + 1]
            )
        )

class LSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size=10000,
        embedding_dim=100,
        hidden_dim=128
    ):
        super(LSTMModel, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)

        return x, hidden

def generate_text(
    model,
    seed_text,
    vocab,
    inv_vocab,
    device,
    length=50,
    temperature=1.0
):
    model.eval()

    words = seed_text.lower().split()
    input_ids = [
        vocab.get(word, vocab["<UNK>"])
        for word in words
    ]

    input_tensor = torch.tensor(
        input_ids
    ).unsqueeze(0).to(device)

    hidden = None

    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(
                input_tensor,
                hidden
            )

            logits = output[0, -1] / temperature
            probs = torch.nn.functional.softmax(
                logits,
                dim=-1
            )

            next_id = torch.multinomial(
                probs,
                num_samples=1
            ).item()

            words.append(
                inv_vocab.get(next_id, "<UNK>")
            )

            # Extend input sequence with new token
            input_ids.append(next_id)
            input_tensor = torch.tensor(
                input_ids
            ).unsqueeze(0).to(device)

    return " ".join(words)