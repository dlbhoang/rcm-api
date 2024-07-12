import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle
import warnings
from helpers import vn_processing as xt  # Assuming vn_processing contains stepByStep function

warnings.filterwarnings('ignore')

# Đọc dữ liệu từ tệp CSV vào DataFrame
data = pd.read_csv('data_cleaned/data_model.csv')

# Chuyển đổi dữ liệu bằng hàm xt.stepByStep

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X = data['Comment Tokenize']
y = data['Label']

# Ánh xạ nhãn chuỗi sang số nguyên
label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
y = y.map(label_map)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo và fit mô hình TF-IDF
tfidf = TfidfVectorizer(
    ngram_range=(1, 3),  # Phạm vi số từ ghép (n-grams) từ 1 đến 3
    min_df=0.02,         # Bỏ qua các từ có tần số xuất hiện thấp hơn 2%
    max_df=0.9           # Bỏ qua các từ có tần số xuất hiện cao hơn 90%
)

tfidf.fit(X_train)

# Lưu trữ mô hình TF-IDF đã được huấn luyện vào tệp tin 'models/tfidf.pkl'
with open('models/tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Biến đổi dữ liệu văn bản thành ma trận TF-IDF
X_train_tfidf = tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Khởi tạo và huấn luyện mô hình XGBoost
model = XGBClassifier(
    objective='multi:softmax',  # Sử dụng softmax cho bài toán phân loại nhiều lớp
    num_class=3,                # Số lớp phân loại
    n_estimators=100,           # Số lượng cây quyết định (estimators)
    max_depth=6,                # Độ sâu tối đa của cây
    learning_rate=0.3,          # Tốc độ học (learning rate)
    random_state=42
)

model.fit(X_train_tfidf, y_train)

# Lưu trữ mô hình XGBoost đã được huấn luyện vào tệp tin 'models/xgboots_model.pkl'
with open('models/xgboots_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Đã lưu mô hình XGBoost vào 'models/xgboots_model.pkl'")

# Đọc mô hình XGBoost từ file
with open('models/xgboots_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Đọc mô hình TF-IDF từ file
with open('models/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Hàm dự đoán nhãn từ một danh sách các đánh giá
def enter_your_comment(text):
    # Chuyển đổi dữ liệu bằng hàm xt.stepByStep
    df = pd.DataFrame({'Comment': text})
    df['Comment Tokenize'] = df['Comment'].apply(xt.stepByStep)
    
    # Transform text to TF-IDF matrix using the same vectorizer as during training
    X_test = tfidf.transform(df['Comment Tokenize'])
    
    # Predict labels
    y_pred = model.predict(X_test)
    
    # Map labels back to original text categories
    df['Label'] = y_pred
    df['Label'] = df['Label'].map({0: 'Negative', 1: 'Neutral', 2: 'Positive'})
    df = df[['Comment', 'Label']]
    return df

# Thử nghiệm một số bình luận
test = ['đồ ăn bình thường, giá hơi cao so với thị trường, chờ quá lâu mới nhận được đơn hàng',
        'hương vị dễ dùng, giá cả hợp lí, giao hàng nhanh, lần sau sẽ ủng hộ',
        'đồ ăn không có gì đặc sắc, nhân viên phục vụ hơi cọc',
        'món thịt xiên nướng ở nhà hàng này khá đặc biệt, thịt mềm và thơm, không bị hôi, nói chung giá cả hợp túi tiền']

# In kết quả dự đoán
print(enter_your_comment(test))
