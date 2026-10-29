from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.bigram_model import BigramModel
import spacy
import io
from pathlib import Path
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from helper_lib.model import get_model
from app.rnn_model import (LSTMModel, generate_text as generate_rnn_text)
from torchvision.utils import make_grid

app = FastAPI()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

# Module 7: load the trained LSTM text generation model.
rnn_checkpoint_path = (
    Path(__file__).resolve().parent
    / "checkpoints_rnn"
    / "rnn_model.pth"
)

rnn_checkpoint = torch.load(
    rnn_checkpoint_path,
    map_location="cpu"
)

rnn_model = LSTMModel()
rnn_model.load_state_dict(
    rnn_checkpoint["model_state_dict"]
)
rnn_model.eval()

rnn_vocab = rnn_checkpoint["vocab"]
rnn_inv_vocab = rnn_checkpoint["inv_vocab"]


@app.post("/generate_with_rnn")
def generate_with_rnn(request: TextGenerationRequest):
    generated_text = generate_rnn_text(
        model=rnn_model,
        seed_text=request.start_word,
        vocab=rnn_vocab,
        inv_vocab=rnn_inv_vocab,
        device="cpu",
        length=request.length
    )

    return {
        "generated_text": generated_text
    }
    
nlp = spacy.load("en_core_web_lg")

class EmbeddingRequest(BaseModel):
    query_word: str


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = nlp(request.query_word).vector

    return {
        "query_word": request.query_word,
        "embedding": embedding.tolist(),
    }

# Assignment 2: load the trained CIFAR-10 classifier.
checkpoint_path = (
    Path(__file__).resolve().parent
    / "checkpoints_assignment2"
    / "best_model.pth"
)

classifier = get_model("AssignmentCNN")
checkpoint = torch.load(checkpoint_path, map_location="cpu")
classifier.load_state_dict(checkpoint["model_state_dict"])
classifier.eval()

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# Use the same preprocessing as during training.
image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        with Image.open(io.BytesIO(contents)) as image:
            image_tensor = image_transform(
                image.convert("RGB")
            ).unsqueeze(0)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    with torch.no_grad():
        logits = classifier(image_tensor)
        class_id = logits.argmax(dim=1).item()

    return {
        "class_id": class_id,
        "class_name": CIFAR10_CLASSES[class_id]
    }

# Assignment 3: load the trained MNIST GAN generator.
assignment3_checkpoint_path = (
    Path(__file__).resolve().parent
    / "checkpoints_assignment3"
    / "assignment3_gan.pth"
)

assignment3_generator, _ = get_model(
    "Assignment3GAN"
)

assignment3_checkpoint = torch.load(
    assignment3_checkpoint_path,
    map_location="cpu"
)

assignment3_generator.load_state_dict(
    assignment3_checkpoint[
        "generator_state_dict"
    ]
)

assignment3_generator.eval()

@app.get("/generate_mnist")
def generate_mnist(num_images: int = 16):
    if num_images < 1 or num_images > 25:
        raise HTTPException(
            status_code=400,
            detail="num_images must be between 1 and 25."
        )

    with torch.no_grad():
        noise = torch.randn(
            num_images,
            100
        )

        generated_images = assignment3_generator(
            noise
        )

    grid = make_grid(
        generated_images,
        nrow=5,
        normalize=True,
        value_range=(-1, 1)
    )

    output_image = transforms.ToPILImage()(
        grid
    )

    image_buffer = io.BytesIO()
    output_image.save(
        image_buffer,
        format="PNG"
    )
    image_buffer.seek(0)

    return StreamingResponse(
        image_buffer,
        media_type="image/png"
    )