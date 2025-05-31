import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ultralytics import YOLO

def apply_pca_enhancement(image, n_components=1, whiten=False, svd_solver='auto'):
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


def main():
    # Khởi tạo mô hình YOLO (đường dẫn tới mô hình của bạn)
    yolo_model = YOLO("best_trash.pt")
    
    # Đường dẫn tới ảnh
    image_path = r"D:\pca\paper218_jpg.rf.d7ce365b331c8165b95dd59c20546654_comp1_orig_annot.png"
    
    # Đọc ảnh gốc (3 kênh BGR)
    orig_color = cv2.imread(image_path)
    if orig_color is None:
        print("Không thể đọc ảnh!")
        return

    # Dự đoán trên ảnh gốc với YOLO
    results_orig = yolo_model.predict(source=image_path,iou=0.3 ,conf=0.2, agnostic_nms=True, max_det=1000)
    # Sử dụng phương thức plot() của kết quả để nhận ảnh đã vẽ bounding box
    annotated_orig = results_orig[0].plot()

    # Áp dụng PCA để tăng cường ảnh (kết quả là ảnh có đủ 3 kênh)
    enhanced_color = apply_pca_enhancement(image_path)
    if enhanced_color is None:
        return
    
    # Dự đoán trên ảnh sau PCA với YOLO
    results_enh = yolo_model.predict(source=enhanced_color,iou=0.3 ,conf=0.2, agnostic_nms=True, max_det=1000)
    annotated_enh = results_enh[0].plot()
    
    # Chuyển ảnh từ BGR sang RGB để hiển thị đúng màu với matplotlib
    annotated_orig_rgb = cv2.cvtColor(annotated_orig, cv2.COLOR_BGR2RGB)
    annotated_enh_rgb = cv2.cvtColor(annotated_enh, cv2.COLOR_BGR2RGB)
    
    # Hiển thị ảnh so sánh
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(annotated_orig_rgb)
    plt.title("YOLO nhận dạng trên ảnh gốc")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(annotated_enh_rgb)
    plt.title("YOLO nhận dạng trên ảnh sau PCA")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
