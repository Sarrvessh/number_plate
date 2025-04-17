if __name__ == "__main__":
    from ultralytics import YOLO

    model = YOLO("yolov8s.pt")

    model.train(
        data="data.yaml",
        epochs=15,
        imgsz=640,
        batch=8,
        project="license_plate_detection",
        name="yolov8_model",
        workers=0, 
    )
