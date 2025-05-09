import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib
from tqdm import tqdm

# Tạo thư mục lưu mô hình
os.makedirs("models", exist_ok=True)

# Load dữ liệu
print("Read dataset")
df = pd.read_csv("url_dataset.csv")  # Gồm cột "url" và "type"
df.dropna(subset=['url', 'type'], inplace=True)

X = df['url'].astype(str)
y = df['type']

# Encode nhãn
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Vector hóa URL
print("Vectorize dataset")
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=100000)
X_vectorized = vectorizer.fit_transform(X)

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,        # Số lượng cây
    max_depth=10,            # Độ sâu của cây
    random_state=42,         # Đảm bảo tính tái lập
    n_jobs=-1                # Sử dụng tất cả các CPU có sẵn
)

# Hiển thị tiến trình huấn luyện với tqdm
print("Training...")
# Tạo một vòng lặp thủ công để huấn luyện và hiển thị tiến trình
for i in tqdm(range(1, rf_model.n_estimators + 1), desc="Training progress", unit="tree"):
    rf_model.set_params(n_estimators=i)  # Điều chỉnh số lượng cây
    rf_model.fit(X_train, y_train)  # Huấn luyện lại với mỗi số cây tăng dần

# Đánh giá mô hình
y_pred = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Lưu mô hình và encoder vào thư mục models/
joblib.dump(rf_model, "models/rf_url_classifier.pkl")
joblib.dump(vectorizer, "models/url_vectorizer.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
print("Mô hình đã được lưu vào thư mục models/")
