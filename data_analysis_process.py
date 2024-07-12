import pandas as pd

# Đọc dữ liệu từ file CSV và loại bỏ các dòng có giá trị null
data = pd.read_csv('data_cleaned/data_analysis.csv').dropna()

# Kiểm tra xem 'Label' có trong danh sách các cột hay không
if 'Label' not in data.columns:
    raise ValueError("Column 'Label' is missing in the dataset.")

# Lấy các cột cần thiết và lưu vào data_model
required_columns = ['UserID', 'User', 'Comment Tokenize', 'Label']
data_model = data[required_columns]

# Lưu dữ liệu mô hình vào file CSV và in thông tin dữ liệu
data_model.to_csv('data_cleaned/data_model.csv', index=False)
print(data_model.info())
