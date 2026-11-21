import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# Same random seeds as the professor's Practical.
torch.manual_seed(42)
np.random.seed(42)

# Check which GPU is available.
# Same device selection as the professor's Practical.
device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

print(f"Using device: {device}")


# Activity requirement: use SQuAD instead of Count of Monte Cristo.
dataset = load_dataset("rajpurkar/squad")

train_data = dataset["train"]
test_data = dataset["validation"]

print(f"Training examples: {len(train_data)}")
print(f"Test examples: {len(test_data)}")


# Inspect one example before creating the DataLoaders.
example = train_data[0]

print("\nContext:")
print(example["context"])

print("\nQuestion:")
print(example["question"])

print("\nAnswer:")
print(example["answers"]["text"][0])

# Use the tokenizer that matches the pretrained GPT-2 model.
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Implementation choices for this activity, not professor-specified values.
MAX_LEN = 512
BATCH_SIZE = 2


class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        context = example["context"]
        question = example["question"]
        answer = example["answers"]["text"][0]

        # Keep the question and answer when shortening long contexts.
        question_answer = (
            f"\nQuestion: {question}\nAnswer: {answer}"
            + tokenizer.eos_token
        )

        question_answer_ids = tokenizer.encode(
            question_answer,
            add_special_tokens=False
        )

        context_ids = tokenizer.encode(
            f"Context: {context}",
            add_special_tokens=False
        )

        # Reserve room for the question and answer.
        if len(question_answer_ids) >= MAX_LEN:
            input_ids = question_answer_ids[:MAX_LEN]
        else:
            context_budget = MAX_LEN - len(question_answer_ids)
            input_ids = (
                context_ids[:context_budget] + question_answer_ids
            )

        attention_mask = [1] * len(input_ids)

        padding_length = MAX_LEN - len(input_ids)

        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(
            attention_mask,
            dtype=torch.long
        )

        labels = input_ids.clone()

        # Ignore padding when calculating the loss.
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


train_dataset = TextDataset(train_data)
test_dataset = TextDataset(test_data)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Inspect one batch, as in the professor's Practical.
batch = next(iter(train_loader))

print("\nInput shape:", batch["input_ids"].shape)
print("Attention mask shape:", batch["attention_mask"].shape)
print("Labels shape:", batch["labels"].shape)

# Load the pretrained model, as specified in the activity.
model = AutoModelForCausalLM.from_pretrained(
    "openai-community/gpt2"
).to(device)

model.config.pad_token_id = tokenizer.pad_token_id

print("\nGPT-2 loaded successfully.")
print(f"Model device: {next(model.parameters()).device}")

from tqdm import tqdm
from pathlib import Path


# Adapted from the professor's train_gpt function.
def train_gpt(model, dataloader, optimizer, epochs, device):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        data_loader_with_progress = tqdm(
            iterable=dataloader,
            ncols=120,
            desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for batch_number, batch in enumerate(data_loader_with_progress):
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)

            optimizer.zero_grad()

            # GPT-2 calculates next-token prediction loss internally.
            outputs = model(
                input_ids=inputs,
                attention_mask=attention_mask,
                labels=targets
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (
                batch_number % 100 == 0
                or batch_number == len(dataloader) - 1
            ):
                data_loader_with_progress.set_postfix(
                    {
                        "avg loss": (
                            f"{total_loss / (batch_number + 1):.4f}"
                        )
                    }
                )


# Initial CPU trial settings, not professor-specified parameters.
# Use a small subset to verify the training and saving workflow.
trial_train_data = train_data.shuffle(seed=42).select(range(200))

trial_train_loader = DataLoader(
    TextDataset(trial_train_data),
    batch_size=BATCH_SIZE,
    shuffle=True
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=5e-5
)

train_gpt(
    model,
    trial_train_loader,
    optimizer,
    epochs=1,
    device=device
)


# Save both the fine-tuned model and its tokenizer for the API.
checkpoint_dir = (
    Path(__file__).resolve().parent
    / "checkpoints_llm"
)

model.save_pretrained(checkpoint_dir)
tokenizer.save_pretrained(checkpoint_dir)

print(f"\nModel and tokenizer saved to: {checkpoint_dir}")