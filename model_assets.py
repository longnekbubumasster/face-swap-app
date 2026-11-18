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
SWAPPER_MIN_BYTES = 100_000_000


def _valid_swapper(path: Path) -> bool:
    # Mirrors can update metadata and therefore change the file checksum.
    # Reject HTML/error responses while allowing compatible ONNX revisions.
    return path.is_file() and path.stat().st_size > SWAPPER_MIN_BYTES


def ensure_swapper_model() -> Path:
    custom_path = os.getenv("INSWAPPER_MODEL_PATH")
    if custom_path:
        path = Path(custom_path).expanduser().resolve()
        if not _valid_swapper(path):
            raise RuntimeError(f"Model INSWAPPER_MODEL_PATH không hợp lệ hoặc chưa tải đủ: {path}")
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
            raise RuntimeError("Model tải về chưa đầy đủ. Vui lòng deploy lại để Render tải lại model.")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination
