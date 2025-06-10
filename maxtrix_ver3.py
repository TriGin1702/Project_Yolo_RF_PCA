# maxtrix3.py
import os
import cv2
import numpy as np
import torch
import joblib
from ultralytics import YOLO
from collections import Counter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score, precision_recall_fscore_support, accuracy_score
# -------------------------------
# Cấu hình ban đầu
# -------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# YOLO_MODEL_PATH = r"best_trash.pt"
# RF_MODEL_PATH   = r"random_forest_model.pkl"
# PCA_SAVE_DIR = r"pca"
YOLO_MODEL_PATH = r"best_garbage.pt"
RF_MODEL_PATH   = r"random_forest_model2_garbage.pkl"
PCA_SAVE_DIR = r"pca_garbage"
# Mapping nhãn – theo huấn luyện RF của bạn
# NAMES_MAP = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'organic', 4: 'paper', 5: 'plastic'}
# CLASS_NAMES = ["cardboard", "glass", "metal", "organic", "paper", "plastic"]
NAMES_MAP = {0: 'Cardboard', 1: 'Garbage', 2: 'Glass', 3: 'Metal', 4: 'Paper', 5: 'Plastic', 6: 'Trash'}
CLASS_NAMES = ["Cardboard", "Garbage", "Glass", "Metal", "Paper", "Plastic", "Trash"]
# Ngưỡng confidence toàn cục của YOLO
GLOBAL_CONF = 0.2

# # Ngưỡng riêng cho một số lớp (theo huấn luyện)
# CLASS_CONFS = {
#     # 0: 0.2,  # cardboard
#     1: 0.2, # glass
#     2: 0.2, # metal
#     # 4: 0.2,
#     # 5: 0.2   # plastic
# }

# RF_WINDOW   = {
#     # 0: 0.4,
#     1: 0.35,
#     2: 0.35,
#     # 4: 0.4,
#     # 5: 0.2,
# }
# Ngưỡng riêng cho một số lớp (theo huấn luyện)
CLASS_CONFS = {
    # 0: 0.2,  # cardboard
    2: 0.2, # glass
    3: 0.2, # metal
    4: 0.2, # paper
    5: 0.2,   # plastic
    6: 0.2   # plastic
}

RF_WINDOW   = {
    # 0: 0.1,
    2: 0.3,
    3: 0.3,
    4: 0.3,
    5: 0.15,
    6: 0.15
}
RF_SCALING  = [0.8, 1.0, 1.2]
IOU_THRESHOLD = 0.3  # ngưỡng IoU khi đối chiếu ground truth với dự đoán

# Thư mục chứa ảnh, nhãn và thư mục để lưu ảnh dự đoán
# IMAGE_FOLDER = r"trash/test/images"
# LABEL_FOLDER = r"trash/test/labels"
# IMAGE_FOLDER = r"/trash_new/test/images"
# LABEL_FOLDER = r"/trash_new/test/labels"
# OUTPUT_FOLDER = r"pre/"
IMAGE_FOLDER = r"garbage/test/images"
LABEL_FOLDER = r"garbage/test/labels"
OUTPUT_FOLDER = r"pre_garbage"
# -------------------------------
# Load mô hình YOLO và RF
# -------------------------------
yolo = YOLO(YOLO_MODEL_PATH)
modules = list(yolo.model.model)
BACKBONE_LAYERS = 7
backbone = torch.nn.Sequential(*modules[:BACKBONE_LAYERS]).to(DEVICE).eval()

rf_model = joblib.load(RF_MODEL_PATH)
# rf_model,lb = joblib.load(RF_MODEL_PATH)
print("Mô hình RF đã được load với tham số:", rf_model.get_params())

# -------------------------------
# Hàm trích xuất đặc trưng
# -------------------------------
# def extract_deep_features(roi):
#     x = cv2.resize(roi, (640, 640))
#     x = x[..., ::-1].copy()  # BGR -> RGB
#     x = x.transpose(2, 0, 1)[None]  # HWC -> CHW
#     x = torch.from_numpy(x).float().to(DEVICE) / 255.0
#     with torch.no_grad():
#         fmap = backbone(x)
#     vec = fmap.mean(dim=[2, 3]).squeeze().cpu().numpy()
#     return vec
def extract_deep_features(roi, layer_indices=None):
    """
    Trích xuất deep-feature từ YOLOv10:
      - hook 3 layer cuối của backbone + Detect module.
      - global‐average pooling và concat thành vector 1D.

    Args:
      - roi (np.ndarray): vùng ảnh HxWx3 (BGR).
      - layer_indices (list[int] or None): index các lớp cần hook.
        Nếu None, mặc định lấy:
          - 3 layer cuối của backbone
          - và layer Detect (module cuối)
    Returns:
      - deep_vec (np.ndarray): vector đặc trưng 1D.
    """
    total = len(yolo.model.model)
    if layer_indices is None:
        # 3 lớp cuối backbone: total-4, total-3, total-2
        # Detect module ở index total-1
        layer_indices = [total-4, total-3, total-2, total-1]

    # 1) Preprocess ROI
    x = cv2.resize(roi, (640, 640))
    x = x[..., ::-1].copy()           # BGR→RGB
    x = x.transpose(2, 0, 1)[None]    # HWC→1xCxHxW
    x = torch.from_numpy(x).float().to(DEVICE) / 255.0

    # 2) Chuẩn bị hook
    features = []
    hooks = []
    def hook_fn(module, inp, out):
        if isinstance(out, torch.Tensor) and out.dim() == 4:
            features.append(out.detach())

    for idx in layer_indices:
        hooks.append(yolo.model.model[idx].register_forward_hook(hook_fn))

    # 3) Forward để kích hoạt hook
    with torch.no_grad():
        _ = yolo.model(x)

    # 4) Remove hook
    for h in hooks:
        h.remove()

    # 5) Global‐avg pooling + concat
    vecs = []
    for fmap in features:
        # mỗi fmap: [1, C, H, W]
        v = fmap.mean(dim=[2,3]).squeeze(0).cpu().numpy()  # → (C,)
        vecs.append(v)
    deep_vec = np.concatenate(vecs, axis=0)               # → (sum_C,)

    return deep_vec


def extract_manual_features(roi, polygon, bg_thresh=20):
    x, y, w, h = cv2.boundingRect(polygon)
    if w == 0 or h == 0:
        return None
    rh, rw = roi.shape[:2]
    corners = np.array([
        roi[0, 0],
        roi[0, rw - 1],
        roi[rh - 1, 0],
        roi[rh - 1, rw - 1]
    ], dtype=np.float32)
    mask = np.ones((rh, rw), dtype=bool)
    for c in corners:
        dist = np.linalg.norm(roi.astype(np.float32) - c, axis=2)
        mask &= (dist > bg_thresh)
    area = float(np.sum(mask))
    area_ratio = area / (w * h) if (w * h) > 0 else 1.0
    aspect_ratio = float(w) / h if h > 0 else 0
    if area > 0:
        b = float(roi[:, :, 0][mask].mean())
        g = float(roi[:, :, 1][mask].mean())
        r = float(roi[:, :, 2][mask].mean())
    else:
        b, g, r = cv2.mean(roi)[:3]
    return [w, h, area_ratio, aspect_ratio, b, g, r]

def apply_pca_enhancement(image, n_components=2, whiten=False, svd_solver='auto'):
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    else:
        img = image.copy()
    if img is None:
        print("Không đọc được ảnh!")
        return None

    img_norm = img.astype(np.float32) / 255.0
    h, w = img_norm.shape[:2]
    chans = 1 if img_norm.ndim == 2 else img_norm.shape[2]
    data = img_norm.reshape(-1, chans)

    # Giảm số chiều theo n_components
    n_comp = min(n_components, chans)
    pca = PCA(n_components=n_comp, whiten=whiten, svd_solver=svd_solver)
    data_pca = pca.fit_transform(data)
    data_rec = pca.inverse_transform(data_pca)

    # Chuyển về ảnh uint8
    if chans > 1:
        img_rec = data_rec.reshape((h, w, chans))
    else:
        img_rec = data_rec.reshape((h, w))
    img_rec = np.clip(img_rec * 255.0, 0, 255).astype(np.uint8)

    # Histogram equalization kênh màu (nếu cần)
    if 1 < chans <= 4:
        img_out = np.zeros_like(img_rec)
        for i in range(chans):
            img_out[..., i] = cv2.equalizeHist(img_rec[..., i])
        return img_out

    return img_rec

def rf_predict_with_scaling(rf_model, features, yolo_label, scales=RF_SCALING):
    preds = []
    for scale in scales:
        scaled = features.copy()
        scaled[0] *= scale
        scaled[1] *= scale
        probs = rf_model.predict_proba([scaled])[0]
        label_idx = int(np.argmax(probs))
        preds.append(label_idx)
    cnt = Counter(preds)
    most_common, count = cnt.most_common(1)[0]
    if count >= 2:
        return most_common
    for idx, name in NAMES_MAP.items():
        if name == yolo_label:
            return idx
    return None

# -------------------------------
# Hàm dự đoán
# -------------------------------

def yolo_only_predict(image, model=yolo, conf_th=GLOBAL_CONF):
    res = model.predict(source=image,iou=0.3 ,conf=conf_th, agnostic_nms=True, max_det=1000, show=False)[0]
    boxes  = res.boxes.xyxy.cpu().numpy()
    cids   = res.boxes.cls.cpu().numpy().astype(int)
    scores = res.boxes.conf.cpu().numpy()
    names = model.model.names if hasattr(model.model, "names") else NAMES_MAP
    return [(cid, box, score, names.get(cid, "unknown")) for cid, box, score in zip(cids, boxes, scores)]



def combined_predict_rf(image):
    """
    Dùng YOLO + RF (nếu score nằm trong cửa sổ) để dự đoán đối tượng.
    Trả về tuple:
      (class_id_YOLO, bounding_box, score, yolo_label, final_label)
    """
    res = yolo.predict(source=image, iou = 0.3,conf=GLOBAL_CONF,agnostic_nms=True, max_det=1000, show=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    class_ids = res.boxes.cls.cpu().numpy().astype(int)
    scores = res.boxes.conf.cpu().numpy()
    names = yolo.model.names if hasattr(yolo.model, "names") else NAMES_MAP
    preds = []
    for box, cid, score in zip(boxes, class_ids, scores):
        yolo_label = names.get(cid, "unknown") if isinstance(names, dict) else names[cid]
        overlap_sum = sum(compute_iou(box, ob) for ob in boxes if not np.array_equal(ob, box))
        use_rf = False
        if cid in CLASS_CONFS and cid in RF_WINDOW:
            th = CLASS_CONFS[cid]
            win = RF_WINDOW[cid]
            if th <= score < (th + win) and (overlap_sum > 0.9 or overlap_sum < 0.1):
                use_rf = True
        final_label = yolo_label
        if use_rf:
            x1, y1, x2, y2 = box.astype(int)
            roi = image[y1:y2, x1:x2]
            if roi is None or roi.size == 0:
                final_label = yolo_label
            else:
                poly = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.int32)
                mfeat = extract_manual_features(roi, poly)
                if mfeat is None:
                    features = extract_deep_features(roi).tolist()
                else:
                    dfeat = extract_deep_features(roi).tolist()
                    features = mfeat + dfeat
                rf_pred_idx = rf_predict_with_scaling(rf_model, features, yolo_label)
                final_label = names.get(rf_pred_idx, yolo_label) if isinstance(names, dict) else names[rf_pred_idx]
        preds.append((cid, box, score, yolo_label, final_label))
    return preds
def combined_predict_rf_pca(image,img_path = ""):
    """
    Dùng YOLO + RF (nếu score nằm trong cửa sổ) để dự đoán đối tượng.
    Trả về tuple:
      (class_id_YOLO, bounding_box, score, yolo_label, final_label)
    """
    
    res = yolo.predict(source=image, iou=0.3,conf=GLOBAL_CONF,agnostic_nms=True, max_det=1000, show=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    class_ids = res.boxes.cls.cpu().numpy().astype(int)
    scores = res.boxes.conf.cpu().numpy()
    names = yolo.model.names if hasattr(yolo.model, "names") else NAMES_MAP
    preds = []
    """
    YOLO + RF (có fallback PCA). Nếu fallback PCA phát hiện được:
      - vẽ khung + nhãn (final_label:confidence) lên cả ảnh gốc & PCA
      - lưu 2 file *_orig_annot.png và *_pca_annot.png
    Trả về preds như thường (cid, box, score, yolo_label, final_label).
    """
    os.makedirs(PCA_SAVE_DIR, exist_ok=True)
    original = image.copy()
    if len(boxes) == 0:
        print("[Fallback] No detections, applying PCA enhancement...")
        os.makedirs(PCA_SAVE_DIR, exist_ok=True)
        original = image.copy()
        detected = False

        # Thử trước comp=2 rồi comp=1
        for comp in [2, 1]:
            print(f"  - Trying PCA with {comp} component(s)")
            enhanced = apply_pca_enhancement(image, n_components=comp)
            if enhanced is None:
                continue

            # Chạy YOLO-only trên ảnh PCA, không refine RF
            res2 = yolo.predict(
                source=enhanced,
                iou=0.3,
                conf=GLOBAL_CONF,
                agnostic_nms=True,
                max_det=1000,
                show=False
            )[0]
            boxes2     = res2.boxes.xyxy.cpu().numpy()
            class_ids2 = res2.boxes.cls.cpu().numpy().astype(int)
            scores2    = res2.boxes.conf.cpu().numpy()

            if len(boxes2) > 0:
                # Annotate ảnh PCA và lưu
                for (x1, y1, x2, y2), cid2, scr in zip(boxes2, class_ids2, scores2):
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    label = names.get(cid2, "unknown")
                    text  = f"{label}:{scr:.2f}"

                    # Vẽ khung đỏ
                    cv2.rectangle(enhanced, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # Tính kích thước text
                    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    # Nền trắng bên trong khung (góc trên–trái)
                    bg_x1 = x1 + 1
                    bg_y1 = y1 + 1
                    bg_x2 = x1 + tw + 3
                    bg_y2 = y1 + th + baseline + 3
                    cv2.rectangle(
                        enhanced,
                        (bg_x1, bg_y1),
                        (bg_x2, bg_y2),
                        (255, 255, 255),
                        thickness=-1
                    )

                    # Ghi text bên trong nền
                    text_x = bg_x1 + 1
                    text_y = bg_y2 - 2
                    cv2.putText(
                        enhanced,
                        text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA
                    )

                # Lưu ảnh gốc annotate và ảnh PCA annotate
                base      = os.path.splitext(os.path.basename(img_path))[0]
                orig_path = os.path.join(PCA_SAVE_DIR, f"{base}_comp{comp}_orig_annot.png")
                pca_path  = os.path.join(PCA_SAVE_DIR, f"{base}_comp{comp}_pca_annot.png")
                cv2.imwrite(orig_path, original)
                cv2.imwrite(pca_path, enhanced)

                # Cập nhật kết quả YOLO-only từ PCA
                boxes     = boxes2
                class_ids = class_ids2
                scores    = scores2
                image     = enhanced
                print(f"    -> Detected with {comp} component(s), using YOLO labels only")
                detected = True
                # Append kết quả vào preds và trả về ngay
                for box2, cid2, scr in zip(boxes2, class_ids2, scores2):
                    yolo_label = names.get(cid2, "unknown")
                    preds.append((cid2, box2, scr, yolo_label, yolo_label))
                return preds
                # break;
        if not detected:
            print("    -> No detections after PCA fallback.")

    for box, cid, score in zip(boxes, class_ids, scores):
        yolo_label = names.get(cid, "unknown") if isinstance(names, dict) else names[cid]
        overlap_sum = sum(compute_iou(box, ob) for ob in boxes if not np.array_equal(ob, box))
        use_rf = False
        if cid in CLASS_CONFS and cid in RF_WINDOW:
            th = CLASS_CONFS[cid]
            win = RF_WINDOW[cid]
            if th <= score < (th + win) and (overlap_sum > 0.9 or overlap_sum < 0.1):
                use_rf = True
        final_label = yolo_label
        if use_rf:
            x1, y1, x2, y2 = box.astype(int)
            roi = image[y1:y2, x1:x2]
            if roi is None or roi.size == 0:
                final_label = yolo_label
            else:
                poly = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.int32)
                mfeat = extract_manual_features(roi, poly)
                if mfeat is None:
                    features = extract_deep_features(roi).tolist()
                else:
                    dfeat = extract_deep_features(roi).tolist()
                    features = mfeat + dfeat
                rf_pred_idx = rf_predict_with_scaling(rf_model, features, yolo_label)
                final_label = names.get(rf_pred_idx, yolo_label) if isinstance(names, dict) else names[rf_pred_idx]
        preds.append((cid, box, score, yolo_label, final_label))
    return preds


# -------------------------------
# Hàm đọc ground truth (YOLO format)
# -------------------------------
def read_ground_truth(label_path, img_w, img_h):
    """
    Đọc file nhãn theo định dạng YOLO (class, x_center, y_center, width_norm, height_norm)
    và chuyển đổi về bounding box pixel.
    Trả về list các tuple: (class_id, [x1, y1, x2, y2])
    """
    gts = []
    if not os.path.exists(label_path):
        return gts
    with open(label_path, 'r') as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) < 5:
                continue
            cid = int(toks[0])
            xc, yc, wn, hn = map(float, toks[1:5])
            x1 = int((xc - wn/2) * img_w)
            y1 = int((yc - hn/2) * img_h)
            x2 = int((xc + wn/2) * img_w)
            y2 = int((yc + hn/2) * img_h)
            gts.append((cid, [x1, y1, x2, y2]))
    return gts

# -------------------------------
# Hàm tính IoU giữa 2 bounding box
# -------------------------------
def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

# -------------------------------
# Hàm đánh giá (YOLO-only và YOLO+RF)
# -------------------------------

def evaluate_predictions():
    y_true = []            # ground truth (chuỗi)
    y_pred_yolo = []       # dự đoán của YOLO-only
    y_pred_rf = []         # dự đoán của YOLO+RF
    y_pred_combine = []    # dự đoán của YOLO+RF+PCA
    correct_yolo = 0
    correct_rf = 0
    correct_pca = 0
    image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_path in image_files:
        image = cv2.imread(img_path)
        if image is None:
            print("Không đọc được ảnh:", img_path)
            continue
        h, w = image.shape[:2]
        label_path = os.path.join(LABEL_FOLDER, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
        gts = read_ground_truth(label_path, w, h)
        if not gts:
            continue

        preds_yolo = yolo_only_predict(image)
        preds_rf   = combined_predict_rf(image)
        preds_combine = combined_predict_rf_pca(image,img_path)


        for gt_c, gt_box in gts:
            best_iou_yolo = 0; best_pred_yolo = None
            best_iou_rf   = 0; best_pred_rf = None
            best_iou_comb = 0; best_pred_comb = None
            gt_label = NAMES_MAP.get(gt_c,'unknown')




            # YOLO-only
            for pred in preds_yolo:
                _, box, score, pred_label = pred
                iou = compute_iou(gt_box, box)
                if iou > best_iou_yolo:
                    best_iou_yolo = iou
                    best_pred_yolo = pred_label

            # YOLO + RF
            for pred in preds_rf:
                _, box, score, yolo_label,final_label = pred
                iou = compute_iou(gt_box, box)
                if iou > best_iou_rf:
                    best_iou_rf = iou
                    best_pred_rf = final_label

            # YOLO + RF + PCA
            for pred in preds_combine:
                _, box, score, yolo_label,final_label = pred
                iou = compute_iou(gt_box, box)
                if iou > best_iou_comb:
                    best_iou_comb = iou
                    best_pred_comb = final_label
            # Đếm đúng nếu IoU đủ ngưỡng và nhãn trùng
            if best_iou_yolo >= IOU_THRESHOLD and best_pred_yolo == gt_label:
                correct_yolo += 1
            if best_iou_rf   >= IOU_THRESHOLD and best_pred_rf   == gt_label:
                correct_rf   += 1
            if best_iou_comb >= IOU_THRESHOLD and best_pred_comb == gt_label:
                correct_pca  += 1
            gt_name = NAMES_MAP.get(gt_c, "unknown")

            # Nếu YOLO phát hiện tốt
            if best_iou_yolo >= IOU_THRESHOLD:
                y_true.append(gt_name)
                y_pred_yolo.append(best_pred_yolo if best_pred_yolo is not None else "unknown")
                y_pred_rf.append(best_pred_rf if best_pred_rf is not None else "unknown")
                y_pred_combine.append(best_pred_comb if best_pred_comb is not None else "unknown")
            # Nếu YOLO không phát hiện, nhưng PCA phát hiện tốt
            elif best_iou_comb >= IOU_THRESHOLD:
                y_true.append(gt_name)
                y_pred_yolo.append("unknown")
                y_pred_rf.append("unknown")
                y_pred_combine.append(best_pred_comb if best_pred_comb is not None else "unknown")
            # Không cái nào đủ tốt thì bỏ qua
            else:
                continue

    class_names = [NAMES_MAP[k] for k in sorted(NAMES_MAP.keys())]

    if not y_true:
        print("Không có ground truth hợp lệ, không thể tính toán Confusion Matrix và các chỉ số đánh giá.")
        return

    print(f"\nTổng số ground truth được đánh giá: {len(y_true)}")
    # Sửa lại đếm mẫu được cứu bởi PCA
    print(f"- Mẫu từ YOLO-only / RF: {sum(1 for y in y_pred_rf if y != 'unknown')}")
    print(f"- Mẫu được cứu bởi PCA: {sum(1 for i, y in enumerate(y_pred_rf) if y == 'unknown' and y_pred_combine[i] != 'unknown')}")
    # for i, (rf_pred, combine_pred) in enumerate(zip(y_pred_rf, y_pred_combine)):
    #     print(f"Index {i}: YOLO+RF = {rf_pred}, YOLO+RF+PCA = {combine_pred}")
    print("\n--- YOLO-only Confusion Matrix ---")
    cm_yolo = confusion_matrix(y_true, y_pred_yolo, labels=class_names)
    print(cm_yolo)
    print(classification_report(y_true, y_pred_yolo, labels=class_names, zero_division=0))

    print("\n--- YOLO+RF Confusion Matrix ---")
    cm_rf = confusion_matrix(y_true, y_pred_rf, labels=class_names)
    print(cm_rf)
    print(classification_report(y_true, y_pred_rf, labels=class_names, zero_division=0))

    print("\n--- YOLO+RF+PCA Confusion Matrix ---")
    cm_combine = confusion_matrix(y_true, y_pred_combine, labels=class_names)
    print(cm_combine)
    print(classification_report(y_true, y_pred_combine, labels=class_names, zero_division=0))
    # print(f"Tổng ground truth: {total_gt}")

    print(f"Accuracy YOLO-only: {correct_yolo}/{len(y_true)} = {correct_yolo/len(y_true):.2%}")
    print(f"Accuracy YOLO+RF:   {correct_rf}/{len(y_true)} = {correct_rf/len(y_true):.2%}")
    print(f"Accuracy YOLO+RF+PCA: {correct_pca}/{len(y_true)} = {correct_pca/len(y_true):.2%}")

# -------------------------------
# Main: dự đoán và lưu ảnh vào OUTPUT_FOLDER
# -------------------------------
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_path in image_files:
        image = cv2.imread(img_path)
        if image is None:
            print("Không đọc được ảnh:", img_path)
            continue

        predictions = combined_predict_rf(image)
        for idx, (cid, box, score, yolo_label, final_label) in enumerate(predictions):
            x1, y1, x2, y2 = box.astype(int)
            # Vẽ khung đối tượng
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)

            # Chuẩn bị text
            text1 = f"yolov10: {yolo_label}"
            text2 = f"rf: {final_label}"

            # Tính kích thước text1
            (tw1, th1), baseline1 = cv2.getTextSize(text1,
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6, 2)
            # Vẽ nền đen mờ (opacity) phía trong đỉnh box
            overlay = image.copy()
            cv2.rectangle(
                overlay,
                (x1, y1),
                (x1 + tw1 + 6, y1 + th1 + baseline1 + 6),
                (0, 0, 0),
                thickness=-1
            )
            # ghép overlay với image để tạo effect trong suốt (tùy chọn)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

            # Vẽ text1 bên trong box, cách viền trên một khoảng 3px
            text_x = x1 + 3
            text_y = y1 + th1 + 3
            cv2.putText(image, text1, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2,
                    lineType=cv2.LINE_AA)

            # Tương tự cho text2, đặt ngay dưới text1
            (tw2, th2), baseline2 = cv2.getTextSize(text2,
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6, 2)
            text2_x = x1 + 3
            text2_y = text_y + th2 + 3
            cv2.putText(image, text2, (text2_x, text2_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2,
                    lineType=cv2.LINE_AA)

        out_path = os.path.join(OUTPUT_FOLDER, os.path.basename(img_path))
        cv2.imwrite(out_path, image)
        print("Đã lưu ảnh dự đoán:", out_path)

    # Sau khi xử lý xong ảnh, thực hiện đánh giá dựa trên ground truth
    # evaluate_predictions()
    evaluate_predictions()