import hashlib
import os
import urllib.request
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(PROJECT_DIR / ".models"))).resolve()
FACE_MODEL = os.getenv("FACE_MODEL", "buffalo_s")
SWAPPER_FILENAME = "inswapper_128.onnx"
SWAPPER_URL = os.getenv(
    "INSWAPPER_MODEL_URL",
    "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
)
SWAPPER_SHA256 = "0fa95f167682b4f61edf24f8d66c46b4ab130e8be058f00c8150e6d0170ca72f"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_swapper(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 100_000_000 and _sha256(path) == SWAPPER_SHA256


def ensure_swapper_model() -> Path:
    custom_path = os.getenv("INSWAPPER_MODEL_PATH")
    if custom_path:
        path = Path(custom_path).expanduser().resolve()
        if not _valid_swapper(path):
            raise RuntimeError(f"Model INSWAPPER_MODEL_PATH không hợp lệ: {path}")
        return path

    # Reuse the user's existing local model without copying another 555 MB file.
    local_path = PROJECT_DIR / SWAPPER_FILENAME
    if _valid_swapper(local_path):
        return local_path

    destination = MODEL_DIR / SWAPPER_FILENAME
    if _valid_swapper(destination):
        return destination

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(SWAPPER_URL, headers={"User-Agent": "face-swap-app/1.0"})
    print(f"Downloading {SWAPPER_FILENAME} to {destination} ...", flush=True)
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        if not _valid_swapper(temporary):
            raise RuntimeError("Model tải về không đúng kích thước hoặc checksum.")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination
