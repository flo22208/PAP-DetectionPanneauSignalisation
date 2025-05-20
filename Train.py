import cv2
from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.yaml")
    results = model.train(data="PAP-DetectionPanneauSignalisation/dataset.yaml", epochs=500, device='cuda')

if __name__ == '__main__':
    main()
