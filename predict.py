import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Tải mô hình, vectorizer và label encoder
rf_model = joblib.load("models/rf_url_classifier.pkl")
vectorizer = joblib.load("models/url_vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Hàm dự đoán với mô hình đã huấn luyện
def predict_url_type(urls):
    X = vectorizer.transform(urls)
    predictions = rf_model.predict(X)
    predicted_labels = label_encoder.inverse_transform(predictions)
    return predicted_labels

# Dữ liệu URL mẫu để dự đoán
urls_to_predict = [str(input('Enter url: '))]

# Dự đoán
predicted_types = predict_url_type(urls_to_predict)
# Hiển thị kết quả dự đoán
for url, prediction in zip(urls_to_predict, predicted_types):
    print(f"URL: {url} => Predicted type: {prediction}")
