import pandas as pd
from pathlib import Path

# --- Paths (adapt if needed) ---
project_root = Path("ppe_object_detection")  # or Path(".") if you're already in that folder

csv_dir       = project_root / "tf_record_files"
img_train_dir = project_root / "ppe_dataset" / "train"
img_test_dir  = project_root / "ppe_dataset" / "test"

yolo_root    = project_root / "yolo_dataset"
images_root  = yolo_root / "images"
labels_root  = yolo_root / "labels"

# Create directory structure
for split in ["train", "val"]:
    (images_root / split).mkdir(parents=True, exist_ok=True)
    (labels_root / split).mkdir(parents=True, exist_ok=True)

# --- Class map (check exact strings in your CSV!) ---
class_map = {
    "Mask": 0,
    "Face Shield": 1,
    "Full Cover": 2,
    "Gloves": 3,
    "Goggles": 4
}

def csv_to_yolo(csv_path, img_dir, split):
    """
    Convert a CSV (train_labels or test_labels) to YOLO .txt files
    and copy images into yolo_dataset/images/<split>.
    """
    df = pd.read_csv(csv_path)
    print(f"Processing {csv_path.name}: {len(df)} rows")

    # Group by filename so each .txt file corresponds to one image
    for filename, group in df.groupby("filename"):
        src_img_path = img_dir / filename
        if not src_img_path.exists():
            # Optional: try other extensions or log missing files
            print(f"WARNING: image not found for {filename} in {img_dir}")
            continue

        # Copy image into YOLO images/<split> (if not already copied)
        dst_img_path = images_root / split / filename
        if not dst_img_path.exists():
            dst_img_path.write_bytes(src_img_path.read_bytes())

        # Use width/height from the CSV (assumed consistent per filename)
        w = group["width"].iloc[0]
        h = group["height"].iloc[0]

        label_lines = []
        for _, row in group.iterrows():
            cls_name = row["class"]
            if cls_name not in class_map:
                print(f"WARNING: class {cls_name} not in class_map, skipping")
                continue

            cls_id = class_map[cls_name]

            xmin, xmax = row["xmin"], row["xmax"]
            ymin, ymax = row["ymin"], row["ymax"]

            # Convert to YOLO format (normalized)
            x_center = ((xmin + xmax) / 2.0) / w
            y_center = ((ymin + ymax) / 2.0) / h
            box_w    = (xmax - xmin) / w
            box_h    = (ymax - ymin) / h

            label_lines.append(
                f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
            )

        # Write YOLO label file
        label_path = labels_root / split / (Path(filename).stem + ".txt")
        label_path.write_text("\n".join(label_lines))

# --- Run for train and val (test) ---
csv_to_yolo(csv_dir / "train_labels.csv", img_train_dir, split="train")
csv_to_yolo(csv_dir / "test_labels.csv",  img_test_dir,  split="val")

print("Done. Check yolo_dataset/images and yolo_dataset/labels.")
