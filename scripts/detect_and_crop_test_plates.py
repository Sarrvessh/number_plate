from pathlib import Path
import cv2
import csv
import re
from ultralytics import YOLO
import easyocr

# Initialize detection model and OCR reader
model = YOLO(r"E:\DATA SCIENTIST_ASSIGNMENT\license_plate_detection\yolov8_model2\weights\best.pt")
ocr_reader = easyocr.Reader(['en'])

# Directory and output list
test_img_dir = "test"
test_imgs = sorted(Path(test_img_dir).glob("*.jpg"), key=lambda x: int(x.stem))
submission_rows = []

def one_hot_encode(digit):
    row = ['0'] * 10
    if digit.isdigit():
        row[int(digit)] = '1'
    return row

# Process each image
for img_path in test_imgs:
    img_id = img_path.stem
    print(f"\nProcessing image: {img_path.name}")

    # Detect license plate
    results = model(img_path)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    print(f"Detected boxes for {img_path.name}: {boxes}")

    if len(boxes) == 0:
        print("No plates detected.")
        continue

    # Crop plate
    x1, y1, x2, y2 = map(int, boxes[0])
    img = cv2.imread(str(img_path))
    plate_crop = img[y1:y2, x1:x2]
    cv2.imwrite(f"cropped_{img_path.name}", plate_crop)

    # OCR recognition
    result = ocr_reader.readtext(plate_crop, detail=0)
    raw_text = ''.join(result)
    digits = re.findall(r'\d', raw_text)

    if len(digits) == 7:
        digits = digits[:3] + digits[-4:]
        print(f"Valid 7-digit plate for {img_path.name}: {digits}")
    else:
        print(f"Digits detected (not 7) for {img_path.name}: {digits}")

    # One-hot encode digits
    for idx, digit in enumerate(digits):
        row_id = f"{img_id}_{idx+1}"
        one_hot = one_hot_encode(digit)
        submission_rows.append([row_id] + one_hot)

# Write to CSV
with open("submissions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id"] + [str(i) for i in range(10)])
    writer.writerows(submission_rows)

print("\nsubmissions.csv created.")
