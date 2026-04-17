from pathlib import Path
import time
import os
import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from tensorflow.keras.models import load_model
import handTracking_module as htm

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parents[1]
MODEL_DIR = PROJECT_ROOT / "model"

model = load_model(MODEL_DIR / "airwriting_model.h5")
class_names = np.load(MODEL_DIR / "class_names.npy", allow_pickle=True)

# Webcam + hand detector
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
detector = htm.handDetector(maxHand=1)

# Drawing state
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
prev_x, prev_y = 0, 0
p_time = 0
last_predict_time = 0
predict_cooldown = 1.2  # seconds

pred_text = ""
pred_conf = 0.0
IMG_SIZE = 28
MIN_FG_PIXELS = 120
MIN_BBOX_SIDE = 32
TOP1_CONF_THRESHOLD = 0.85
TOP1_TOP2_MARGIN = 0.20

def preprocess_canvas_for_model(canvas_img):
    gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th_bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fg_bin = cv2.countNonZero(th_bin)
    fg_inv = cv2.countNonZero(th_inv)
    bin_img = th_bin if fg_bin <= fg_inv else th_inv

    kernel = np.ones((2, 2), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

    ys, xs = np.where(bin_img > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    fg_pixels = int(cv2.countNonZero(bin_img))
    bbox_w = int(x_max - x_min + 1)
    bbox_h = int(y_max - y_min + 1)

    if fg_pixels < MIN_FG_PIXELS:
        return None
    if bbox_w < MIN_BBOX_SIDE or bbox_h < MIN_BBOX_SIDE:
        return None

    crop = bin_img[y_min:y_max + 1, x_min:x_max + 1]

    h, w = crop.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.uint8)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = crop

    pad = max(2, int(side * 0.2))
    square = cv2.copyMakeBorder(square, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    square = cv2.resize(square, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    x = square.astype(np.float32) / 255.0
    x = x.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return x

def run_prediction(canvas_img):
    x = preprocess_canvas_for_model(canvas_img)
    if x is None:
        return "UNCERTAIN", 0.0

    # Light test-time augmentation to improve stability on noisy input.
    aug_batch = [x]
    aug_batch.append(np.roll(x, shift=1, axis=1))
    aug_batch.append(np.roll(x, shift=-1, axis=1))
    aug_batch.append(np.roll(x, shift=1, axis=2))
    aug_batch.append(np.roll(x, shift=-1, axis=2))

    probs = np.mean([model.predict(sample, verbose=0)[0] for sample in aug_batch], axis=0)
    top2_idx = np.argsort(probs)[-2:][::-1]
    idx = int(top2_idx[0])
    second_idx = int(top2_idx[1])

    top1_conf = float(probs[idx])
    top2_conf = float(probs[second_idx])
    margin = top1_conf - top2_conf

    if top1_conf < TOP1_CONF_THRESHOLD or margin < TOP1_TOP2_MARGIN:
        return "UNCERTAIN", top1_conf

    return str(class_names[idx]), top1_conf

while True:
    success, frame = cap.read()
    if not success:
        print("Cannot read webcam frame.")
        break

    frame = cv2.flip(frame, 1)

    frame = detector.findHands(frame)
    lm_list, _ = detector.findPositions(frame)
    fingers = detector.fingersUp()

    drawing_mode = False

    if len(lm_list) != 0:
        x1, y1 = lm_list[8][1], lm_list[8][2]  # index fingertip

        # Draw when only index finger is up
        if fingers == [0, 1, 0, 0, 0]:
            drawing_mode = True
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x1, y1
                continue
            smooth_x = int(prev_x * 0.7 + x1 * 0.3) if prev_x != 0 else x1
            smooth_y = int(prev_y * 0.7 + y1 * 0.3) if prev_y != 0 else y1
            cv2.line(canvas, (prev_x, prev_y), (smooth_x, smooth_y), (255, 255, 255), 8)
            prev_x, prev_y = smooth_x, smooth_y

        # Erase mode when index + middle are up and close
        elif fingers == [0, 1, 1, 0, 0]:
            d = ((lm_list[8][1] - lm_list[12][1]) ** 2 + (lm_list[8][2] - lm_list[12][2]) ** 2) ** 0.5
            mx = (lm_list[8][1] + lm_list[12][1]) // 2
            my = (lm_list[8][2] + lm_list[12][2]) // 2
            if d < 40:
                cv2.circle(frame, (mx, my), 28, (125, 200, 255), cv2.FILLED)
                cv2.circle(canvas, (mx, my), 28, (0, 0, 0), cv2.FILLED)
            prev_x, prev_y = 0, 0

        # Clear canvas with all fingers down (fist-like)
        elif fingers == [0, 0, 0, 0, 0]:
            canvas = np.zeros_like(canvas)
            pred_text = ""
            pred_conf = 0.0
            prev_x, prev_y = 0, 0

        # Predict when only thumb is up
        elif fingers[0] == 1 and sum(fingers[1:]) == 0:
            if (time.time() - last_predict_time) > predict_cooldown and np.sum(canvas) > 0:
                pred_text, pred_conf = run_prediction(canvas)
                print(f"Prediction: {pred_text}  Confidence: {pred_conf:.4f}")
                last_predict_time = time.time()
            prev_x, prev_y = 0, 0

        else:
            prev_x, prev_y = 0, 0

    if not drawing_mode and len(lm_list) == 0:
        prev_x, prev_y = 0, 0

    # FPS
    c_time = time.time()
    fps = 1.0 / (c_time - p_time) if p_time != 0 else 0.0
    p_time = c_time

    # Overlay drawing canvas
    display = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0.0)

    # UI text
    cv2.putText(display, f"FPS: {int(fps)}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display, "Draw: index finger", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(display, "Predict: thumb up", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(display, "Clear: fist", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(display, "Quit: q", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if pred_text:
        color = (0, 255, 0) if pred_text != "UNCERTAIN" else (0, 165, 255)
        cv2.putText(display, f"Pred: {pred_text}", (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(display, f"Conf: {pred_conf:.2%}", (20, 225),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Air Writing Prediction", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()