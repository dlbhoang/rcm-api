import pandas as pd
import helpers.vn_processing as xt

# Đọc dữ liệu từ tệp CSV 'review.csv' vào DataFrame reviews
reviews = pd.read_csv('data/review.csv')

# Hiển thị các dòng đầu tiên của DataFrame reviews
print(reviews.head())

# Hiển thị thông tin tổng quan về DataFrame reviews
print(reviews.info())

# Loại bỏ các hàng có giá trị thiếu trong cột 'Comment' của DataFrame reviews
reviews = reviews.dropna(subset=['Comment'])

# Hiển thị thông tin tổng quan về DataFrame reviews sau khi loại bỏ giá trị thiếu
print(reviews.info())

# Loại bỏ các hàng có giá trị thiếu trong cột 'User' của DataFrame reviews
reviews = reviews.dropna(subset=['User'])

# Hiển thị thông tin tổng quan về DataFrame reviews sau khi loại bỏ giá trị thiếu
print(reviews.info())

# Xóa cột 'Unnamed: 0' khỏi DataFrame reviews
reviews = reviews.drop(columns=['Unnamed: 0'])

# Hiển thị các dòng đầu tiên của DataFrame reviews sau khi đã xử lý
print(reviews.head())

# Chuyển đổi kiểu dữ liệu của cột 'UserID' và 'RestaurantID' sang kiểu object (chuỗi)
reviews['UserID'] = reviews['UserID'].astype('str')
reviews['RestaurantID'] = reviews['RestaurantID'].astype('str')

# Hiển thị thông tin tổng quan về DataFrame reviews sau khi chuyển đổi kiểu dữ liệu
print(reviews.info())

# Kiểm tra dữ liệu trùng lặp trong DataFrame reviews
print(f"Số lượng dữ liệu trùng lặp trong DataFrame reviews: {reviews.duplicated().sum()}")

# Đếm số lượng giá trị trùng lặp trong cột 'UserID' của DataFrame reviews
print(f"Số lượng giá trị trùng lặp trong cột 'UserID': {reviews['UserID'].duplicated().sum()}")

# Đếm số lượng giá trị duy nhất trong cột 'RestaurantID' của DataFrame reviews
print(f"Số lượng giá trị duy nhất trong cột 'RestaurantID': {reviews['RestaurantID'].nunique()}")

# Tính toán các thống kê mô tả cho cột 'Rating' trong DataFrame reviews
print(reviews['Rating'].describe())

# Nhóm và đếm số lượng đánh giá theo từng mức 'Rating' trong DataFrame reviews
rating = reviews.groupby('Rating')['User'].count().reset_index()

# Sắp xếp lại DataFrame rating theo cột 'Rating' giảm dần
rating = rating.sort_values(by='Rating', ascending=False)

# Hiển thị các dòng đầu tiên của DataFrame rating
print(rating.head())

# Áp dụng hàm xt.stepByStep để tách từ trong cột 'Comment' của DataFrame reviews
reviews['Comment Tokenize'] = reviews['Comment'].apply(xt.stepByStep)

# Hiển thị các dòng đầu tiên của DataFrame reviews sau khi áp dụng hàm để tokenize cột 'Comment'
print(reviews.head())

# Lưu DataFrame reviews đã được làm sạch vào tệp CSV 'review_cleaned.csv'
reviews.to_csv('data_cleaned/review_cleaned.csv', index=False)
