import io
import os
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.predict import FashionClassifier

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best_model.pth"))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
classifier: FashionClassifier | None = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global classifier
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model checkpoint not found: {MODEL_PATH}")
    classifier = FashionClassifier(MODEL_PATH)
    yield
    classifier = None


app = FastAPI(title="Fashion Classifier API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if classifier is not None else "unavailable"}


@app.get("/classes")
def classes() -> dict[str, list[str]]:
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"classes": classifier.class_names}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image")
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MiB or smaller")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(contents)) as image:
                image.verify()
            with Image.open(io.BytesIO(contents)) as image:
                image.load()
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise HTTPException(status_code=400, detail="Invalid or unsupported image") from error

    if classifier is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    with Image.open(io.BytesIO(contents)) as image:
        return classifier.predict(image)
