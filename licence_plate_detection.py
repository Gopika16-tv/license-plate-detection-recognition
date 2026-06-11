"""
License Plate Detection & Recognition
=======================================
Tuned for: top-down highway/road camera footage (car_video.mp4)
Python   : 3.10 / 3.14
Usage    : python license_plate_detector.py

Controls:
  Q     = Quit
  S     = Save screenshot
  SPACE = Pause / Resume
  +/-   = Speed up / slow down playback
"""

import cv2
import pytesseract
import numpy as np
import re
from datetime import datetime

# ── Windows Tesseract path ────────────────────────────────
# Set Tesseract path here if required for your system

# ── Video file ────────────────────────────────────────────
VIDEO_FILE = "input_video.mp4"   # must be in the same folder as this script


# ─────────────────────────────────────────────────────────
# Detect moving objects (cars) using background subtraction
# This works well for fixed cameras like this highway cam
# ─────────────────────────────────────────────────────────
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=200, varThreshold=50, detectShadows=False
)


def detect_cars(frame):
    """Use background subtraction to find moving cars in the frame."""
    fg_mask = bg_subtractor.apply(frame)

    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_h, frame_w = frame.shape[:2]
    car_regions = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800:        # ignore tiny noise
            continue
        if area > frame_w * frame_h * 0.5:  # ignore huge blobs
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Expand bounding box slightly to capture full car
        pad = 10
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(frame_w - x, w + pad * 2)
        h = min(frame_h - y, h + pad * 2)

        car_regions.append((x, y, w, h))

    return car_regions, fg_mask


# ─────────────────────────────────────────────────────────
# Inside each car region, search for plate-like sub-regions
# ─────────────────────────────────────────────────────────
def find_plate_in_car(car_crop):
    """
    Given a cropped car image, find the license plate sub-region.
    Returns list of (x, y, w, h) relative to the crop.
    """
    gray = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(blurred, 30, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    ch, cw = car_crop.shape[:2]
    plates = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        aspect = w / h
        # Plate aspect ratio: 2x to 7x wide
        if not (1.8 <= aspect <= 7.5):
            continue
        # Minimum size
        if w < 30 or h < 8:
            continue
        # Not too big relative to car crop
        if w > cw * 0.95 or h > ch * 0.6:
            continue
        plates.append((x, y, w, h))

    return plates


# ─────────────────────────────────────────────────────────
# Preprocess plate crop for Tesseract OCR
# ─────────────────────────────────────────────────────────
def preprocess_for_ocr(plate_img):
    h, w = plate_img.shape[:2]
    if h < 5 or w < 10:
        return None

    # Upscale — larger image = better OCR
    scale = max(4, int(80 / h))
    resized = cv2.resize(plate_img, (w * scale, h * scale),
                         interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=15)

    # CLAHE — improves contrast (good for night footage)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(denoised)

    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5.5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # Threshold
    _, binary = cv2.threshold(sharpened, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # White border helps Tesseract
    bordered = cv2.copyMakeBorder(binary, 15, 15, 15, 15,
                                  cv2.BORDER_CONSTANT, value=255)
    return bordered


# ─────────────────────────────────────────────────────────
# Run OCR and return best text result
# ─────────────────────────────────────────────────────────
def ocr_plate(plate_img):
    processed = preprocess_for_ocr(plate_img)
    if processed is None:
        return ""

    whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    best = ""

    for psm in [7, 8, 6, 13]:
        config = (
            f"--psm {psm} "
            f"--oem 3 "
            f"-c tessedit_char_whitelist={whitelist}"
        )
        try:
            raw = pytesseract.image_to_string(processed, config=config)
            cleaned = re.sub(r"[^A-Z0-9]", "", raw.upper()).strip()
            if len(cleaned) > len(best):
                best = cleaned
        except Exception:
            continue

    return best


# ─────────────────────────────────────────────────────────
# Validate: must look like a real plate
# ─────────────────────────────────────────────────────────
def is_valid_plate(text):
    if len(text) < 4 or len(text) > 12:
        return False
    has_letter = any(c.isalpha() for c in text)
    has_digit  = any(c.isdigit() for c in text)
    return has_letter and has_digit


# ─────────────────────────────────────────────────────────
# Draw everything on the display frame
# ─────────────────────────────────────────────────────────
def draw_results(frame, car_boxes, plate_results, fps, frame_num, total_frames, all_plates):
    h_frame, w_frame = frame.shape[:2]

    # ── Header bar ────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w_frame, 52), (15, 15, 15), -1)
    cv2.putText(frame, "LICENSE PLATE DETECTOR", (8, 34),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS:{fps:.1f}  {frame_num}/{total_frames}",
                (w_frame - 220, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1, cv2.LINE_AA)

    # ── Progress bar ──────────────────────────────────────
    if total_frames > 0:
        prog = int((frame_num / total_frames) * w_frame)
        cv2.rectangle(frame, (0, h_frame - 7), (w_frame, h_frame), (30, 30, 30), -1)
        cv2.rectangle(frame, (0, h_frame - 7), (prog, h_frame), (0, 200, 100), -1)

    # ── Car boxes (blue outline) ──────────────────────────
    for (x, y, w, h) in car_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 100, 0), 1)

    # ── Plate boxes + text (green) ────────────────────────
    for (x, y, w, h), text in plate_results:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 80), 2)

        # Corner accents
        L = 10
        for cx, cy, dx, dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
            cv2.line(frame, (cx, cy), (cx + dx*L, cy), (0, 255, 255), 2)
            cv2.line(frame, (cx, cy), (cx, cy + dy*L), (0, 255, 255), 2)

        # Text label
        label = f" {text} "
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
        ly = y - 8 if y - 8 > lh + 6 else y + h + lh + 8
        cv2.rectangle(frame, (x, ly - lh - 4), (x + lw, ly + 4), (0, 160, 40), -1)
        cv2.putText(frame, label, (x, ly),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Plate log panel (right side) ─────────────────────
    if all_plates:
        panel_x = w_frame - 180
        cv2.rectangle(frame, (panel_x - 5, 55), (w_frame, 55 + len(all_plates) * 22 + 10),
                      (15, 15, 15), -1)
        cv2.putText(frame, "DETECTED PLATES:", (panel_x, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1, cv2.LINE_AA)
        for i, p in enumerate(sorted(all_plates)[-8:]):  # show last 8
            cv2.putText(frame, f"  {p}", (panel_x, 90 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 80), 1, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def run():
    cap = cv2.VideoCapture(VIDEO_FILE)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open '{VIDEO_FILE}'")
        print("[TIP]   Make sure car_video.mp4 is in the SAME FOLDER as this script.")
        print("[TIP]   If extension differs, change VIDEO_FILE at top of script.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    delay        = int(1000 / video_fps)      # ms per frame

    print(f"[INFO] Video : {VIDEO_FILE}")
    print(f"[INFO] Frames: {total_frames}  |  FPS: {video_fps:.1f}")
    print("[INFO] Controls: Q=quit  S=screenshot  SPACE=pause  +/-=speed")

    frame_num       = 0
    skip            = 2          # process every 2nd frame
    fps             = 0.0
    fps_start       = datetime.now()
    last_results    = []
    last_car_boxes  = []
    paused          = False
    all_plates      = set()
    playback_delay  = delay

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n[INFO] Video finished.")
                break

            frame_num += 1

            # FPS counter
            elapsed = (datetime.now() - fps_start).total_seconds()
            if elapsed >= 1.0:
                fps = frame_num / elapsed

            # ── Detection every N frames ──────────────────
            if frame_num % skip == 0:
                car_boxes, _ = detect_cars(frame)
                last_car_boxes = car_boxes
                last_results   = []

                for (cx, cy, cw, ch) in car_boxes:
                    car_crop = frame[cy:cy+ch, cx:cx+cw]
                    if car_crop.size == 0:
                        continue

                    plate_candidates = find_plate_in_car(car_crop)

                    for (px, py, pw, ph) in plate_candidates:
                        plate_crop = car_crop[py:py+ph, px:px+pw]
                        if plate_crop.size == 0:
                            continue

                        text = ocr_plate(plate_crop)

                        if is_valid_plate(text):
                            # Convert coords back to full-frame
                            abs_x = cx + px
                            abs_y = cy + py
                            last_results.append(((abs_x, abs_y, pw, ph), text))

                            if text not in all_plates:
                                all_plates.add(text)
                                print(f"[FRAME {frame_num:05d}] NEW plate: {text}")

        # ── Draw & display ────────────────────────────────
        display = draw_results(
            frame.copy(), last_car_boxes, last_results,
            fps, frame_num, total_frames, all_plates
        )
        cv2.imshow("License Plate Detector  [Q=quit | S=screenshot | SPACE=pause | +/-=speed]",
                   display)

        key = cv2.waitKey(max(1, playback_delay)) & 0xFF

        if key == ord("q"):
            print("[INFO] Quit.")
            break
        elif key == ord("s"):
            fname = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fname, display)
            print(f"[SAVED] {fname}")
        elif key == ord(" "):
            paused = not paused
            print("[PAUSED]" if paused else "[RESUMED]")
        elif key == ord("+") or key == ord("="):
            playback_delay = max(1, playback_delay - 5)
            print(f"[SPEED] delay={playback_delay}ms")
        elif key == ord("-"):
            playback_delay = min(200, playback_delay + 5)
            print(f"[SPEED] delay={playback_delay}ms")

    cap.release()
    cv2.destroyAllWindows()

    # ── Final summary ─────────────────────────────────────
    print("\n" + "=" * 40)
    print(f"  DONE — {len(all_plates)} unique plate(s) found")
    for p in sorted(all_plates):
        print(f"    {p}")
    print("=" * 40)


if __name__ == "__main__":
    run()
