from ultralytics import YOLO

def main():
    # ----------------------------
    # CONFIG
    # ----------------------------
    DATA_YAML = r"D:\yolov11\rock paper scissor.v9i.yolov11\data.yaml"
    PRETRAINED = "yolo11s.pt"
    EPOCHS = 60
    IMG_SIZE = 612
    DEVICE = "cuda"
    BATCH = 4 

    RUN_NAME = "ex"   # <-- AUTO: ex, ex2, ex3 ...

    # ----------------------------
    # LOAD MODEL
    # ----------------------------
    print(f"Loading model: {PRETRAINED}")
    model = YOLO(PRETRAINED)

    # ----------------------------
    # TRAIN
    # ----------------------------
    print("\nStarting training...")

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        project="runs/train",
        name=RUN_NAME,
        exist_ok=False   # <-- REQUIRED for auto numbering
    )

    # ----------------------------
    # DONE
    # ----------------------------
    best_path = str(results.save_dir) + "/weights/best.pt"

    print("\nTraining Finished!")
    print(f"✔ Model folder: {results.save_dir}")
    print(f"✔ Best weights: {best_path}")

if __name__ == "__main__":
    main()
