import os
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

# --- 1) Load YOLOv10 detection model as backbone extractor ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo = YOLO(r"best_trash.pt")
# Extract modules list and pick backbone layers
modules = list(yolo.model.model)
BACKBONE_LAYERS = 7  # adjust based on model depth
backbone = torch.nn.Sequential(*modules[:BACKBONE_LAYERS]).to(DEVICE).eval()

# --- 2) Deep feature extraction from YOLO backbone ---
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

# --- 3) Manual feature extraction ---
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

# --- 4) Parse YOLO-format or polygon labels ---
def parse_polygon_line(line, img_w, img_h):
    parts = line.strip().split()
    cid = int(parts[0])
    coords = list(map(float, parts[1:]))
    if len(coords) == 4:
        x_c, y_c, w_n, h_n = coords
        x1 = int((x_c - w_n/2)*img_w); y1 = int((y_c - h_n/2)*img_h)
        x2 = int((x_c + w_n/2)*img_w); y2 = int((y_c + h_n/2)*img_h)
        poly = np.array([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]], dtype=np.int32)
        return cid, poly
    pts = []
    for i in range(0, len(coords), 2):
        pts.append([int(coords[i]*img_w), int(coords[i+1]*img_h)])
    return cid, np.array([pts], dtype=np.int32)

# --- 5) Process dataset and save CSV ---
def process_dataset(image_dir, label_dir, names_map, output_csv):
    all_feats = []
    for fn in os.listdir(image_dir):
        if not fn.lower().endswith(('.jpg','png','jpeg')):
            continue
        img_path = os.path.join(image_dir, fn)
        lbl_path = os.path.join(label_dir, os.path.splitext(fn)[0] + '.txt')
        img = cv2.imread(img_path)
        if img is None or not os.path.exists(lbl_path):
            continue
        h, w = img.shape[:2]
        with open(lbl_path) as f:
            for line in f:
                cid, poly = parse_polygon_line(line, w, h)
                if poly is None:
                    continue
                x, y, ww, hh = cv2.boundingRect(poly)
                roi = img[y:y+hh, x:x+ww]
                mfeat = extract_manual_features(roi, poly)
                if mfeat is None:
                    continue
                dfeat = extract_deep_features(roi)
                row = mfeat + dfeat.tolist() + [names_map.get(cid, cid)]
                all_feats.append(row)
    manual_cols = ['width','height','area_ratio','aspect_ratio','avg_b','avg_g','avg_r']
    deep_cols = [f'deep_{i}' for i in range(dfeat.shape[0])]
    df = pd.DataFrame(all_feats, columns=manual_cols + deep_cols + ['label'])
    df.to_csv(output_csv, index=False)
    print(f"Saved features to {output_csv}")

if __name__ == '__main__':
    IMAGE_DIR  = r"trash/train/images"
    LABEL_DIR  = r"trash/train/labels"
    OUTPUT_CSV = r"combined_features.csv"
    NAMES_MAP  = {0:'cardboard',1:'glass',2:'metal',3:'organic',4:'paper',5:'plastic'}
    process_dataset(IMAGE_DIR, LABEL_DIR, NAMES_MAP, OUTPUT_CSV)
