import pandas as pd
from pathlib import Path

# Paths (adapt to your actual layout)
project_root = Path("ppe_bject_detection")  # your repo root
csv_dir = project_root / "tf_record_files"
img_train_dir = project_root / "ppe_dataset" / "train"
img_test_dir  = project_root / "ppe_dataset" / "test"

labels_root = project_root / "yolo_dataset" / "labels"
images_root = project_root / "yolo_dataset" / "images"

(labels_root / "train").mkdir(parents=True, exist_ok=True)
(labels_root / "val").mkdir(parents=True, exist_ok=True)
(images_root / "train").mkdir(parents=True, exist_ok=True)
(images_root / "val").mkdir(parents=True, exist_ok=True)

# Define your class map
class_map = {
    "Mask": 0,
    "Face Shield": 1,
    "Full Cover": 2,
    "Gloves": 3,
    "Goggles": 4
}

# Load train labels
train_df = pd.read_csv(csv_dir / "train_labels.csv")

# Optional: decide what to use as val. Simple approach:
# - Keep Kaggle "train" as YOLO train
# - Use Kaggle "test" as YOLO val
# For now, just build train from train_df.

for filename, group in train_df.groupby("filename"):
    # find the image
    src_img_path = img_train_dir / filename
    if not src_img_path.exists():
        # If not found in train, maybe check test dir if needed
        continue

    # Copy image into YOLO train directory
    dst_img_path = images_root / "train" / filename
    if not dst_img_path.exists():
        dst_img_path.write_bytes(src_img_path.read_bytes())

    # Use width/height from CSV (assumes consistent per filename)
    w = group["width"].iloc[0]
    h = group["height"].iloc[0]

    label_lines = []
    for _, row in group.iterrows():
        cls_name = row["class"]
        if cls_name not in class_map:
            continue
        cls_id = class_map[cls_name]

        xmin, xmax = row["xmin"], row["xmax"]
        ymin, ymax = row["ymin"], row["ymax"]

        x_center = ((xmin + xmax) / 2.0) / w
        y_center = ((ymin + ymax) / 2.0) / h
        box_w    = (xmax - xmin) / w
        box_h    = (ymax - ymin) / h

        label_lines.append(
            f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
        )

    # Write YOLO label file
    label_path = labels_root / "train" / (Path(filename).stem + ".txt")
    label_path.write_text("\n".join(label_lines))
