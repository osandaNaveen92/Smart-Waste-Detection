from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    DATA_YAML = r"C:\Users\Acer\Documents\ET\GARBAGE-CLASSIFICATION-3-2\data.yaml"

    model = YOLO("yolov8m.pt")

    results = model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=418,
        batch=4,
        device=0,
        workers=0,      # ⬅️ set to 0 on Windows to avoid multiprocessing issues
        patience=10,
        save=True,
        project="outputs",
        name="garbage_v1",
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        amp=False,
        verbose=True
    )

    print("\n✅ Training complete!")
    print(f"Best model saved at: outputs/garbage_v1/weights/best.pt")