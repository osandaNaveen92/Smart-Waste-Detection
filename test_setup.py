import torch
import cv2
import numpy as np
from ultralytics import YOLO

print("=== Environment Check ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")

# Test YOLOv8 loads correctly
print("\nLoading YOLOv8...")
model = YOLO("yolov8m.pt")  # auto-downloads ~50MB
print("YOLOv8 loaded successfully ✅")

print("\n✅ All dependencies ready!")