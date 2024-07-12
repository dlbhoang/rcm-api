# Import các thư viện cần thiết
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
restaurants = pd.read_csv('data/restaurant.csv')

# Kiểm tra và xử lý dữ liệu trùng lặp
restaurants = restaurants.drop_duplicates()
restaurants = restaurants.dropna(subset=['Restaurant Name'])

# Tách dữ liệu để lấy quận từ địa chỉ
restaurants['District'] = restaurants['Address'].str.split(', ').apply(lambda x: x[-2] if len(x) > 1 else None)

# Xóa cột không cần thiết
restaurants = restaurants.drop(columns='Unnamed: 0')

# Xử lý dữ liệu về thời gian hoạt động của nhà hàng
restaurants['Time'] = restaurants['Time'].fillna('00:00 - 23:59')
restaurants['Time Open'] = restaurants['Time'].apply(lambda x: x[:5])
restaurants['Time Close'] = restaurants['Time'].apply(lambda x: x[-5:])

# Xử lý dữ liệu về giá của món ăn
restaurants['Lowest Price'] = restaurants['Price'].str.split(' - ').str[0].str.replace(".", "").astype('float')
restaurants['Highest Price'] = restaurants['Price'].str.split(' - ').str[1].str.replace(".", "").astype('float')

# Xử lý dữ liệu các giá trị giảm nhỏ hơn 1000
restaurants.loc[restaurants['Lowest Price'] < 1000, 'Lowest Price'] = 20000
restaurants.loc[restaurants['Highest Price'] < 1000, 'Highest Price'] = 50000

# Sắp xếp lại DataFrame theo RestaurantID
restaurants = restaurants.sort_values(by='RestaurantID')

# Chuyển đổi kiểu dữ liệu của RestaurantID sang object (chuỗi)
restaurants['RestaurantID'] = restaurants['RestaurantID'].astype(str)

# Xóa cột Price và lưu vào file CSV
restaurants = restaurants.drop(columns='Price')
restaurants.to_csv('data_cleaned/restaurant_cleaned.csv', index=False)

# Hiển thị một số dữ liệu và biểu đồ thống kê
print("Thông tin sau khi xử lý:")
print(restaurants.info())
print("\nMột số mẫu dữ liệu:")
print(restaurants.head())

# Tính giá trị trung vị của Lowest Price và Highest Price
lowest_price_median = restaurants['Lowest Price'].median()
highest_price_median = restaurants['Highest Price'].median()

print(f"\nTrung vị của Giá thấp nhất: {lowest_price_median}")
print(f"Trung vị của Giá cao nhất: {highest_price_median}")


