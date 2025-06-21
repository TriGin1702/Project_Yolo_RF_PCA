import os
import cv2
import numpy as np
import torch
import joblib
from ultralytics import YOLO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# -------------------------------
# Cấu hình ban đầu
# -------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
YOLO_MODEL_PATH = r"best_pkl.pt"
IMAGE_FOLDER    = r"pkl/train/images"
LABEL_FOLDER    = r"pkl/train/labels"
NAMES_MAP = {0:'space-empty', 1:'space-occupied'}


# Resize về 320 để giảm feature map size
INPUT_SIZE = 640

# Load YOLO và backbone
yolo = YOLO(YOLO_MODEL_PATH)
modules = list(yolo.model.model)
backbone = torch.nn.Sequential(*modules[:]).to(DEVICE).eval()  # hook ngay cả Detect

def extract_deep_features(roi):
    # 1 layer cuối (Detect, index = -1)
    x = cv2.resize(roi, (INPUT_SIZE, INPUT_SIZE))
    x = x[..., ::-1].transpose(2,0,1)[None]
    x = torch.from_numpy(x).float().to(DEVICE)/255.0

    features = []
    def hook_fn(module, inp, out):
        if isinstance(out, torch.Tensor) and out.dim()==4:
            features.append(out.detach())
    h =yolo.model.model[-1].register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = yolo.model(x)
    h.remove()

    # pooling
    fmap = features[0]  # [1, C, H, W]
    vec = fmap.mean(dim=[2,3]).squeeze(0).cpu().numpy()
    return vec

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

def build_dataset(image_folder, label_folder, names_map):
    X, y = [], []
    for fn in os.listdir(image_folder):
        if not fn.lower().endswith(('.jpg','.png','jpeg')):
            continue
        img_path = os.path.join(image_folder, fn)
        lbl_path = os.path.join(label_folder, os.path.splitext(fn)[0] + '.txt')
        img = cv2.imread(img_path)
        if img is None or not os.path.exists(lbl_path):
            continue
        h, w = img.shape[:2]

        # Đọc toàn bộ lines của file nhãn
        with open(lbl_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        # Chọn 9 ROI: 5 đầu + 4 cuối (nếu đủ), ngược lại lấy hết
        if len(lines) >= 20:
            selected = lines[:10] + lines[-10:]
        else:
            selected = lines

        # Duyệt qua các dòng đã chọn
        for line in selected:
            cid, poly = parse_polygon_line(line, w, h)
            if poly is None:
                continue

            x0, y0, ww, hh = cv2.boundingRect(poly)
            roi = img[y0:y0+hh, x0:x0+ww]
            if roi.size == 0:
                continue

            # Trích manual + deep feature
            mfeat = extract_manual_features(roi, poly)
            dfeat = extract_deep_features(roi)
            feat = (mfeat or []) + dfeat.tolist()

            X.append(feat)
            y.append(names_map.get(cid, cid))

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    return X, y


def train_rf(X,y):
    param_grid = {
        'n_estimators': [700],        # giảm số cây
        'max_depth':   [None],
        'max_samples': [0.7],         # chỉ dùng 50% mẫu mỗi cây
    }
    rf = RandomForestClassifier(random_state=17, class_weight="balanced", n_jobs=-1)
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=27)
    gs = GridSearchCV(rf, param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
    gs.fit(X,y)
    print("Best params:", gs.best_params_)
    print("Best CV acc:", gs.best_score_)
    joblib.dump(gs.best_estimator_, "random_forest_model_pkl.pkl")
    print("Saved model random_forest_model_pkl.pkl")

if __name__=="__main__":
    X, y = build_dataset(IMAGE_FOLDER, LABEL_FOLDER, NAMES_MAP)
    print("Loaded samples:", X.shape)
    train_rf(X,y)
