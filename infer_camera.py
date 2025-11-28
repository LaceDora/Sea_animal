import cv2
import tensorflow as tf
import numpy as np
import json
import time

# Load model và nhãn
model = tf.keras.models.load_model("seafood_model.keras")
with open("labels.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.7   # cao hơn để tránh đoán mò
ROI_SCALE = 1.0              # lấy toàn khung hình để kiểm tra
SHOW_ROI = True              # hiển thị ROI để debug

# Hàm dự đoán
def predict_image(frame):
    # Chuyển sang RGB vì OpenCV đọc BGR
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_array = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    preds = model.predict(img_array, verbose=0)
    #preds = tf.nn.softmax(preds, axis=-1).numpy()[0]

    class_idx = np.argmax(preds)
    confidence = np.max(preds)

    if confidence < CONFIDENCE_THRESHOLD:
        label = "Unknown"
    else:
        label = class_names[class_idx]

    return label, confidence

# Kiểm tra GPU
if not tf.config.list_physical_devices('GPU'):
    print("Running on CPU (no GPU detected)")

# Mở camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở camera.")
    exit()

prev_time = time.time()
frame_count = 0
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    box_size = int(min(h, w) * ROI_SCALE)
    x1 = max(0, w // 2 - box_size // 2)
    y1 = max(0, h // 2 - box_size // 2)
    x2 = min(w, x1 + box_size)
    y2 = min(h, y1 + box_size)
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        roi = frame

    label, conf = predict_image(roi)

    # Cập nhật FPS
    frame_count += 1
    if frame_count >= 10:
        curr_time = time.time()
        fps = 10 / (curr_time - prev_time)
        prev_time = curr_time
        frame_count = 0

    # Hiển thị kết quả
    box_color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    text = f"{label} ({conf*100:.1f}%)"
    cv2.putText(frame, text, (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Camera Feed", frame)

    # Hiển thị ROI để debug
    if SHOW_ROI:
        roi_resized = cv2.resize(roi, (224, 224))
        cv2.imshow("ROI (vùng mô hình đang nhận diện)", roi_resized)

    # Thoát bằng phím Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
