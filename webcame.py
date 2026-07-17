import cv2
import time
from ultralytics import YOLO

def main():
    # ----------------------------
    # CONFIG
    # ----------------------------
    MODEL_PATH = r"D:\yolov11\runs\train\ex28\weights\best.pt"
    #D:\yolov11\runs\train\rps_train_20251207_235454\weights\best.pt
    DEVICE = "cuda"  # or "cpu"
    WEBCAM_ID = 0

    # ----------------------------
    # LOAD MODEL
    # ----------------------------
    print("Loading model:", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    # ----------------------------
    # OPEN WEBCAM
    # ----------------------------
    cap = cv2.VideoCapture(WEBCAM_ID)

    if not cap.isOpened():
        print("❌ Failed to open webcam!")
        return

    print("Webcam opened successfully.")
    print("Press 'q' to quit.\n")

    # Set webcam resolution (optional)
    cap.set(3, 1280)  # width
    cap.set(4, 720)   # height

    # For FPS calculation
    prev_time = time.time()

    # ----------------------------
    # DETECTION LOOP
    # ----------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        # YOLO detection
        results = model(frame, device=DEVICE, conf=0.5)

        # Get annotated frame
        annotated_frame = results[0].plot()

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Draw FPS on screen
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Show result
        cv2.imshow("YOLOv11 Webcam", annotated_frame)

        # Exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
