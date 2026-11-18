"""Download model assets during the Render build instead of storing them in Git."""

from insightface.app import FaceAnalysis

from model_assets import FACE_MODEL, MODEL_DIR, ensure_swapper_model


def main() -> None:
    swapper_path = ensure_swapper_model()
    print(f"Swapper model ready: {swapper_path}")

    # Constructing FaceAnalysis downloads and validates the selected model pack.
    FaceAnalysis(
        name=FACE_MODEL,
        root=str(MODEL_DIR),
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    print(f"InsightFace model ready: {FACE_MODEL}")


if __name__ == "__main__":
    main()
