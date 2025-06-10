import os
import cv2
import numpy as np
import torch
import joblib
from ultralytics import YOLO
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# -------------------------------
# Cấu hình ban đầu
# -------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# YOLO_MODEL_PATH = r"best_trash.pt"
YOLO_MODEL_PATH = r"best_garbage.pt"
# Mapping nhãn theo dự án của bạn
# NAMES_MAP = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'organic', 4: 'paper', 5: 'plastic'}
NAMES_MAP = {0: 'Cardboard', 1: 'Garbage', 2: 'Glass', 3: 'Metal', 4: 'Paper', 5: 'Plastic', 6: 'Trash'}
# GLOBAL_CONF = 0.2
# RF_SCALING = [0.7, 1.0, 1.3]
# IOU_THRESHOLD = 0.3

# IMAGE_FOLDER = r"trash/train/images"
# LABEL_FOLDER = r"trash/train/labels"
IMAGE_FOLDER = r"garbage/train/images"
LABEL_FOLDER = r"garbage/train/labels"
# -------------------------------
# Load mô hình YOLO và lấy backbone (chẳng hạn 7 layer đầu)
# -------------------------------
yolo = YOLO(YOLO_MODEL_PATH)
modules = list(yolo.model.model)
BACKBONE_LAYERS = 7
backbone = torch.nn.Sequential(*modules[:BACKBONE_LAYERS]).to(DEVICE).eval()

# Load RF model nếu bạn muốn dùng RF đã huấn luyện trước (ở đây ta sẽ huấn luyện mới)
# rf_model, lb = joblib.load(RF_MODEL_PATH)
# print("Mô hình RF đã được load với tham số:", rf_model.get_params())

# -------------------------------
# Hàm trích xuất đặc trưng (deep) từ ROI
# -------------------------------
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




# Hàm trích xuất đặc trưng thủ công (ví dụ shape và các thông số cơ bản)
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

# Ví dụ: kết hợp manual và deep features
def extract_features(roi, polygon):
    mfeat = extract_manual_features(roi, polygon)
    dfeat = extract_deep_features(roi)
    if mfeat is None:
        return dfeat  # hoặc chỉ dùng deep features
    else:
        return mfeat + dfeat.tolist()  # kết hợp thành vector một chiều

# -------------------------------
# Hàm parse polygon từ file nhãn (định dạng YOLO)
# -------------------------------
def parse_polygon_line(line, img_w, img_h):
    parts = line.strip().split()
    if len(parts) < 5:
        return None, None
    try:
        cid = int(parts[0])
    except:
        return None, None
    coords = list(map(float, parts[1:]))
    if len(coords) == 4:
        x_c, y_c, wn, hn = coords
        x1 = int((x_c - wn/2) * img_w)
        y1 = int((y_c - hn/2) * img_h)
        x2 = int((x_c + wn/2) * img_w)
        y2 = int((y_c + hn/2) * img_h)
        poly = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.int32)
        return cid, poly
    # Nếu là polygon nhiều điểm:
    pts = []
    for i in range(0, len(coords), 2):
        pts.append([int(coords[i] * img_w), int(coords[i+1] * img_h)])
    poly = np.array([pts], dtype=np.int32)
    return cid, poly

# -------------------------------
# Hàm build dataset từ ảnh và nhãn trực tiếp (trong bộ nhớ)
# -------------------------------
def build_dataset(image_folder, label_folder, names_map):
    X_list = []  # danh sách vector đặc trưng (sẽ là list các list hoặc numpy array)
    y_list = []  # danh sách nhãn (có thể là chuỗi hoặc số)
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for fn in image_files:
        img_path = os.path.join(image_folder, fn)
        lbl_path = os.path.join(label_folder, os.path.splitext(fn)[0] + '.txt')
        img = cv2.imread(img_path)
        if img is None or not os.path.exists(lbl_path):
            continue
        h, w = img.shape[:2]
        with open(lbl_path, 'r', encoding='utf-8') as f:
            for line in f:
                cid, poly = parse_polygon_line(line, w, h)
                if poly is None:
                    continue
                x, y, ww, hh = cv2.boundingRect(poly)
                roi = img[y:y+hh, x:x+ww]
                if roi.size == 0:
                    continue
                features = extract_features(roi, poly)
                X_list.append(features)  # features là vector 1D
                y_list.append(names_map.get(cid, cid))
    # Chuyển thành numpy array với kích thước (n_samples, feature_dim)
    X = np.vstack([np.array(f).reshape(1, -1) for f in X_list])
    y = np.array(y_list)
    return X, y

# -------------------------------
# Huấn luyện RandomForest trực tiếp từ bộ dữ liệu trong bộ nhớ
# -------------------------------
def train_rf(X, y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    import joblib
    
    # Sử dụng một GridSearch đơn giản
    n_samples, n_features = X.shape
    param_grid = {
        'n_estimators': [int(n_samples/2), int(n_samples/1.5)],
        'max_depth': [None],
        'max_samples': [0.8]
    }
    
    rf = RandomForestClassifier(random_state=17, class_weight="balanced")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=27)
    grid_search = GridSearchCV(rf, param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    
    print("Tham số tối ưu:", grid_search.best_params_)
    print("Accuracy trung bình:", grid_search.best_score_)
    
    # Lưu mô hình RF
    # joblib.dump(grid_search.best_estimator_, r"random_forest_model2.pkl")
    # print("Mô hình RandomForest đã được lưu tại: random_forest_model2.pkl")
    joblib.dump(grid_search.best_estimator_, r"random_forest_model2_garbage.pkl")
    print("Mô hình RandomForest đã được lưu tại: random_forest_model2_garbage.pkl")
if __name__ == '__main__':
    # Xây dựng dataset trực tiếp từ ảnh và nhãn (không cần ghi CSV)
    X, y = build_dataset(IMAGE_FOLDER, LABEL_FOLDER, NAMES_MAP)
    print("Số mẫu:", X.shape[0], "với số đặc trưng mỗi mẫu:", X.shape[1])
    # Huấn luyện RF trực tiếp trên dữ liệu trong bộ nhớ
    train_rf(X, y)
