import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

DATA_URL = "https://github.com/alexeygrigorev/clothing-dataset-small/archive/refs/heads/master.zip"
ARCHIVE_ROOT = "clothing-dataset-small-master"
SPLITS = {"train": "train", "validation": "val", "test": "test"}


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        member_path = PurePosixPath(member.filename)
        windows_path = PureWindowsPath(member.filename)
        if (
            member_path.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or ".." in member_path.parts
            or ".." in windows_path.parts
        ):
            raise ValueError(f"Unsafe path in dataset archive: {member.filename}")
        target = (destination / Path(*member_path.parts)).resolve()
        if root not in target.parents and target != root:
            raise ValueError(f"Unsafe path in dataset archive: {member.filename}")
    archive.extractall(destination)


def main() -> None:
    destination = Path("data")
    extracted = destination / ARCHIVE_ROOT
    if all((destination / split).is_dir() for split in ("train", "val", "test")):
        print("Dataset splits already exist in data/")
        return

    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "dataset.zip"
    request = urllib.request.Request(DATA_URL, headers={"User-Agent": "fashion-classifier"})
    try:
        print(f"Downloading dataset from {DATA_URL}")
        with urllib.request.urlopen(request, timeout=60) as response, archive_path.open("wb") as file:
            shutil.copyfileobj(response, file)
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, destination)
    finally:
        archive_path.unlink(missing_ok=True)

    if not extracted.is_dir():
        raise FileNotFoundError(f"Expected dataset folder was not found: {extracted}")
    for source_name, target_name in SPLITS.items():
        source = extracted / source_name
        if not source.is_dir():
            raise FileNotFoundError(f"Dataset split is missing: {source}")
        target = destination / target_name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    shutil.rmtree(extracted)
    print("Dataset ready in data/train, data/val, and data/test")


if __name__ == "__main__":
    main()
