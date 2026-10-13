from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.bigram_model import BigramModel
import spacy
import io
from pathlib import Path
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from helper_lib.model import get_model

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