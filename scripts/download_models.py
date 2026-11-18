"""Download model assets during the Render build instead of storing them in Git."""

from insightface.app import FaceAnalysis

from model_assets import FACE_MODEL, MODEL_DIR


def main() -> None:
    # The local pixel-transplant mode needs only face detection. It deliberately
    # does not download the 555 MB generative swapper model.
    FaceAnalysis(
        name=FACE_MODEL,
        root=str(MODEL_DIR),
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"],
    )
    print(f"InsightFace model ready: {FACE_MODEL}")


if __name__ == "__main__":
    main()
