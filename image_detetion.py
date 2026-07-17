from ultralytics import YOLO
import cv2

def main():
    # ----------------------------
    # CONFIG
    # ----------------------------
    MODEL_PATH = "D:\\yolov11\\runs\\train\\ex28\\weights\\best.pt"           # Or your custom model: "runs/train/exp/weights/best.pt"
    IMAGE_PATH = "photo_3_2026-01-21_11-46-32.jpg"             # Path to the image you want to detect
    DEVICE = "cuda"                     # "cuda" or "cpu"

    # ----------------------------
    # LOAD MODEL
    # ----------------------------
    print("Loading model:", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    # ----------------------------
    # PREDICT
    # ----------------------------
    print("Running detection...")
    results = model.predict(
        source=IMAGE_PATH,
        device=DEVICE,
        save=True,            # Save output image
        save_txt=True,        # Save labels
        conf=0.25             # Confidence threshold
    )

    # ----------------------------
    # DISPLAY RESULT
    # ----------------------------
    result_image_path = results[0].save_dir / results[0].path.name
    print("\nDetection complete!")
    print("Saved output to:", result_image_path)

    # Show the result with OpenCV
    output_img = cv2.imread(str(result_image_path))
    cv2.imshow("YOLOv11 Result", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
