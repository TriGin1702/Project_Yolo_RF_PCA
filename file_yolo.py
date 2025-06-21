# yolo_fruit_test.py
import os
import cv2
import torch
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report

# -------------------------------
# Cấu hình
# -------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
YOLO_MODEL_PATH = "best_fruit.pt"  # đường dẫn tới file YOLO checkpoint
data_folder = "fruit/test/images"  # thư mục ảnh test
label_folder = "fruit/test/labels"  # thư mục label YOLO-format
output_folder = "fruit/pre_yolo"  # thư mục lưu ảnh dự đoán
iou_threshold = 0.3
conf_threshold = 0.2

# Tải model YOLO
yolo = YOLO(YOLO_MODEL_PATH)
yolo.model.to(DEVICE).eval()

# Hàm dự đoán YOLO-only
def yolo_only_predict(image_path):
    img = cv2.imread(image_path)
    res = yolo.predict(source=img, iou=iou_threshold, conf=conf_threshold, agnostic_nms=True, max_det=1000)[0]
    boxes = res.boxes.xyxy.cpu().numpy().astype(int)
    class_ids = res.boxes.cls.cpu().numpy().astype(int)
    scores = res.boxes.conf.cpu().numpy()
    names = yolo.model.names
    preds = []
    for cid, box, score in zip(class_ids, boxes, scores):
        label = names[cid]
        preds.append((cid, box, score, label))
    return preds

# Hàm đọc ground truth
def read_ground_truth(label_path, img_w, img_h):
    gts = []
    if not os.path.exists(label_path):
        return gts
    with open(label_path, 'r') as f:
        for line in f:
            cid, xc, yc, wn, hn = map(float, line.split())
            x1 = int((xc - wn/2) * img_w)
            y1 = int((yc - hn/2) * img_h)
            x2 = int((xc + wn/2) * img_w)
            y2 = int((yc + hn/2) * img_h)
            gts.append((int(cid), [x1, y1, x2, y2]))
    return gts

# Compute IoU
def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (area1 + area2 - inter) if (area1+area2-inter)>0 else 0

# Đánh giá YOLO-only
def evaluate_yolo():
    y_true, y_pred = [], []
    images = [f for f in os.listdir(data_folder) if f.lower().endswith(('.jpg','.png'))]
    for img_name in images:
        img_path = os.path.join(data_folder, img_name)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        label_path = os.path.join(label_folder, os.path.splitext(img_name)[0] + '.txt')
        gts = read_ground_truth(label_path, w, h)
        preds = yolo_only_predict(img_path)

        # Match each GT with best pred
        for cid_true, gt_box in gts:
            best_iou, best_label = 0, None
            for cid, box, score, label in preds:
                iou = compute_iou(gt_box, box)
                if iou > best_iou:
                    best_iou, best_label = iou, label
            if best_iou >= iou_threshold:
                y_true.append(cid_true)
                # map label back to class id
                y_pred.append(list(yolo.model.names.keys())[list(yolo.model.names.values()).index(best_label)])

    # Metrics
    if not y_true:
        print("Không có ground truth để đánh giá.")
        return
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_true, y_pred))

# Main
if __name__ == '__main__':
    os.makedirs(output_folder, exist_ok=True)
    # Chạy đánh giá
    evaluate_yolo()
    # Vẽ và lưu ảnh prediction
    for img_name in os.listdir(data_folder):
        if not img_name.lower().endswith(('.jpg','.png')): continue
        img_path = os.path.join(data_folder, img_name)
        img = cv2.imread(img_path)
        preds = yolo_only_predict(img_path)
        for cid, box, score, label in preds:
            x1,y1,x2,y2 = box
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, f"{label}:{score:.2f}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        out_path = os.path.join(output_folder, img_name)
        cv2.imwrite(out_path, img)
    print("✅ Lưu xong ảnh dự đoán YOLO-only tại:", output_folder)
