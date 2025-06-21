# rf_mispredictions_analysis.py
import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, Counter

# ===============================
# Config
# ===============================
MODEL_PATH = "best_trash.pt"
IMG_DIR    = "trash/test/images"
LABEL_DIR  = "trash/test/labels"
IOU_THRESH = 0.3
CONF_THRESH= 0.2

# ===============================
# Utilities
# ===============================
def compute_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter/(a1 + a2 - inter) if inter>0 else 0

def read_gt(label_path, w, h):
    gt = []
    if os.path.isfile(label_path):
        with open(label_path) as f:
            for line in f:
                c, xc, yc, wn, hn = map(float, line.split()[:5])
                x1 = int((xc - wn/2)*w)
                y1 = int((yc - hn/2)*h)
                x2 = int((xc + wn/2)*w)
                y2 = int((yc + hn/2)*h)
                gt.append((int(c), [x1, y1, x2, y2]))
    return gt

# ===============================
# Step 1: Collect YOLO detections and match GT
# ===============================
yolo = YOLO(MODEL_PATH).eval()
predictions = []  # list of dicts: box, pred, conf, true

for img_file in os.listdir(IMG_DIR):
    if not img_file.lower().endswith(('.jpg','.png','.jpeg')):
        continue
    img_path = os.path.join(IMG_DIR, img_file)
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    label_path = os.path.join(LABEL_DIR, os.path.splitext(img_file)[0] + '.txt')
    gts = read_gt(label_path, w, h)

    # YOLO inference
    res = yolo.predict(source=img, iou=IOU_THRESH, conf=CONF_THRESH, max_det=1000)[0]
    boxes = res.boxes.xyxy.cpu().numpy().astype(int)
    preds = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()

    for box, p, cf in zip(boxes, preds, confs):
        # match to best GT
        best_iou, best_j = 0, None
        for j, (_, gt_box) in enumerate(gts):
            iou_val = compute_iou(box, gt_box)
            if iou_val > best_iou:
                best_iou, best_j = iou_val, j
        true_c = gts[best_j][0] if best_iou >= IOU_THRESH else -1
        predictions.append({
            'box': box,
            'pred': int(p),
            'conf': float(cf),
            'true': int(true_c)
        })

# ===============================
# Step 2: Analyze mispredictions per predicted class
# ===============================
mis_info = defaultdict(list)
for det in predictions:
    p, true_c, cf = det['pred'], det['true'], det['conf']
    if true_c >= 0 and p != true_c:
        mis_info[p].append((cf, true_c))

print("\n=== YOLO Mispredictions Analysis ===")
for p in sorted(mis_info.keys()):
    entries = sorted(mis_info[p], key=lambda x: x[0])
    total_wrong = len(entries)
    print(f"\nPredicted class {p}: total wrong = {total_wrong}")
    # group by confidence (2 decimal bins)
    conf_groups = defaultdict(list)
    for cf, tc in entries:
        key = round(cf, 2)
        conf_groups[key].append(tc)
    # output sorted
    for conf_key in sorted(conf_groups.keys()):
        actuals = conf_groups[conf_key]
        wrong_count = len(actuals)
        # total preds at this conf for class p
        total_at_conf = sum(
            1 for det in predictions
            if det['pred']==p and round(det['conf'],2)==conf_key
        )
        error_rate = wrong_count / total_at_conf if total_at_conf>0 else 0
        true_counts = Counter(actuals)
        confusion_str = ", ".join(f"{tc}:{cnt}" for tc, cnt in true_counts.items())
        print(f"  conf={conf_key:.2f} -> wrong {wrong_count}/{total_at_conf} = {error_rate:.2%} -> actual: {confusion_str}")
    min_conf = min(cf for cf, _ in entries)
    max_conf = max(cf for cf, _ in entries)
    total_preds_in_range = sum(
        1 for det in predictions
        if det['pred'] == p and min_conf <= det['conf'] <= max_conf
    )
    error_rate_range = total_wrong / total_preds_in_range if total_preds_in_range > 0 else 0
    print(f"  --> Conf range [{min_conf:.2f} ~ {max_conf:.2f}] => error rate in range: {error_rate_range:.2%}")


# - "Predicted class 5" nghĩa là YOLO đã đoán class là 5.
# - "actual" tức là class thật (ground truth) mà YOLO đã dự đoán sai.
# - Ví dụ: "conf=0.21 -> wrong 1/3 = 33.33% -> actual: 3:1" nghĩa là: