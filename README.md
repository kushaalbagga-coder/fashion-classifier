# Fashion Classifier

An end-to-end image classifier for clothing and fashion products. It fine-tunes an ImageNet-pretrained EfficientNet-B0 (or ResNet-50), saves the dataset's class names with the model, and serves predictions through FastAPI or Streamlit.

## Requirements

- Python 3.10–3.13
- CPU or CUDA-capable PyTorch installation

Create and activate a virtual environment, then install the project:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For CUDA training, install the PyTorch and torchvision builds appropriate for your CUDA version from the [PyTorch installation selector](https://pytorch.org/get-started/locally/) before installing the remaining project dependencies.

## Dataset and training

Download and arrange the public [Clothing Dataset Small](https://github.com/alexeygrigorev/clothing-dataset-small) into `data/train`, `data/val`, and `data/test`:

```powershell
python scripts\download_data.py
```

The loader uses the folders under `data/train` as class labels and verifies that validation and test splits contain the same classes. Image files are not included in this repository. Review the upstream dataset's terms before use.

Edit `configs\config.yaml` to adjust the model, image size, batch size, epochs, and training schedule, then train:

```powershell
python scripts\train.py --config configs\config.yaml
```

Training starts with a frozen feature extractor and trains the classification head. It then unfreezes the backbone and fine-tunes the complete model. The best validation checkpoint is written to `models\best_model.pth`; it includes the model architecture, class names, and preprocessing image size.

Evaluate the saved checkpoint against the test split:

```powershell
python scripts\evaluate.py --config configs\config.yaml
```

## Run predictions

Start the API after training:

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to try the endpoints. `GET /health` reports readiness, `GET /classes` lists trained labels, and `POST /predict` accepts a JPEG, PNG, or WebP image in the `file` field. Set `MODEL_PATH` to use a checkpoint outside the default `models\best_model.pth` path.

Run the interactive UI in another terminal:

```powershell
streamlit run app\streamlit_app.py
```

The interface includes a sample gallery using the downloaded test images, as well as an upload workspace for side-by-side comparison of up to four images. Use the sidebar to choose how many likely categories to show, set a low-confidence indicator, hide previews, or toggle the comparison chart. Download a CSV style report from each result.

The Python API can also be used directly:

```python
from PIL import Image
from src.predict import FashionClassifier

classifier = FashionClassifier("models/best_model.pth")
result = classifier.predict(Image.open("product.jpg"))
print(result)
```

## Docker

Build and run the API image after training a model locally:

```powershell
docker compose up --build
```

The Compose file mounts the local `models` directory into the container. The model checkpoint is intentionally excluded from Git; train one locally or provide it through your deployment's artifact/storage mechanism.

## Development

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

The test suite covers model construction, checkpoint metadata and prediction, dataset split validation, and API input handling.
