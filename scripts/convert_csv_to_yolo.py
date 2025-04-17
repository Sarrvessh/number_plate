import os
import pandas as pd
import cv2

# Paths
csv_path = "data/Licplatesdetection_train.csv"
image_dir = "data/license_plates_detection_train"
yolo_label_dir = "labels_yolo"

os.makedirs(yolo_label_dir, exist_ok=True)

# Read CSV
df = pd.read_csv(csv_path)

for index, row in df.iterrows():
    img_path = os.path.join(image_dir, row['img_id'])
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Normalize YOLO format
    x_center = ((row['xmin'] + row['xmax']) / 2) / w
    y_center = ((row['ymin'] + row['ymax']) / 2) / h
    box_width = (row['xmax'] - row['xmin']) / w
    box_height = (row['ymax'] - row['ymin']) / h

    # Save label
    label_path = os.path.join(yolo_label_dir, row['img_id'].replace('.jpg', '.txt'))
    with open(label_path, 'w') as f:
        f.write(f"0 {x_center} {y_center} {box_width} {box_height}")
