

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd
import joblib
from sklearn.preprocessing import LabelBinarizer

# Đọc dữ liệu từ file CSV mới chứa các cột: width, height, aspect_ratio, avg_b, avg_g, avg_r, label, set
df = pd.read_csv("combined_features.csv")
# df = pd.read_csv("shape_texture_features.csv")
print("Các cột trong CSV:", df.columns.tolist())

num_labels = df['label'].nunique()
print("Số lượng nhãn:", num_labels)
num_rows = df.shape[0]
print("Số dòng trong file CSV:", num_rows)

# Các cột thuộc tính: tất cả trừ 'label' và 'set'
# feature_columns = df.drop(['label', 'set'], axis=1)
feature_columns = df.drop(['label'], axis=1)
print("Các cột thuộc tính:", feature_columns.columns.tolist())
num_features = len(feature_columns.columns)
# Tăng trọng số cho cột aspect_ratio (ví dụ: nhân với 3.0)
aspect_ratio_weight = 1
feature_columns['aspect_ratio'] = feature_columns['aspect_ratio'] * aspect_ratio_weight
# # Thêm cột bản sao của aspect_ratio để mô hình ưu tiên đặc trưng này hơn
# feature_columns['aspect_ratio_dup'] = feature_columns['aspect_ratio']
# print("Các cột thuộc tính sau khi nhấn mạnh aspect_ratio:", feature_columns.columns.tolist())
# Tạo đặc trưng đầu vào X và nhãn y
X = feature_columns.values
y = df['label'].values

# Tạo đối tượng LabelBinarizer (nếu cần)
lb = LabelBinarizer()
lb.fit(y)
param_grid = {
    
    'n_estimators': [int(num_rows/2),int(num_rows/1.5)],
    'max_depth': [None], #26.04 trung binh, 37 max => 28, 30 , 32
    'max_samples': [0.8]
}
# Khởi tạo mô hình RandomForest với class_weight để xử lý mất cân bằng
rf = RandomForestClassifier(random_state=17, class_weight="balanced")

# Sử dụng StratifiedKFold cho cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=27)

# Thiết lập GridSearchCV-
grid_search = GridSearchCV(rf, param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)

print("Tham số tối ưu:", grid_search.best_params_)
print("Accuracy trung bình:", grid_search.best_score_)

# Lưu mô hình tốt nhất cùng đối tượng LabelBinarizer
joblib.dump((grid_search.best_estimator_, lb), "random_forest_model.pkl")
print("Mô hình RandomForest đã được lưu tại: random_forest_model.pkl")
