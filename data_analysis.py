import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import seaborn as sb

# Đọc dữ liệu từ file CSV vào DataFrame
restaurants = pd.read_csv('data_cleaned/restaurant_cleaned.csv')
reviews = pd.read_csv('data_cleaned/review_cleaned.csv')

# Merge hai DataFrame restaurants và reviews dựa trên cột 'RestaurantID'
data = pd.merge(restaurants, reviews, how='inner', left_on='RestaurantID', right_on='RestaurantID')

# Xóa cột 'Comment' khỏi DataFrame data sau khi merge
data = data.drop(columns={'Comment'})

# Loại bỏ các dòng có giá trị NaN trong cột 'Comment Tokenize'
data.dropna(subset=['Comment Tokenize'], inplace=True)

# Lấy cột 'Comment Tokenize'
text = data['Comment Tokenize']

# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Thiết lập các tham số
tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    min_df=0.02,
    max_df=0.9
)

# Fit vectorizer
tfidf.fit(text)

# Chuyển đổi dữ liệu
text = tfidf.transform(text)

# Tạo DataFrame df_text
df_text = pd.DataFrame(text.toarray(), columns=tfidf.get_feature_names_out())

# Tạo DataFrame hiển thị số lần xuất hiện của mỗi từ khóa trong DataFrame df_text
lst_value = []
lst_name = []

# Lặp qua từng cột trong df_text
for col in df_text.columns:
    # Đếm số lần xuất hiện của các giá trị lớn hơn 0 trong từng cột và tính tổng
    value = df_text[df_text[col] > 0][col].value_counts().sum()
    lst_value.append(value)
    lst_name.append(col)

# Tạo DataFrame word_important
word_important = pd.DataFrame({'KeyWord': lst_name, 'Count': lst_value})

# Sắp xếp DataFrame word_important theo cột 'Count' giảm dần
word_important.sort_values(by='Count', ascending=False, inplace=True)

# Tạo danh sách các từ khóa thuộc nhóm Positive
positive_words = [
    "thích", "tốt", "xuất sắc", "tuyệt vời", "tuyệt hảo", "đẹp", "ổn", "ngon",
    "hài lòng", "ưng ý", "hoàn hảo", "chất lượng", "thú vị", "nhanh",
    "tiện lợi", "dễ sử dụng", "hiệu quả", "ấn tượng",
    "nổi bật", "tận hưởng", "tốn ít thời gian", "thân thiện", "hấp dẫn",
    "gợi cảm", "tươi mới", "lạ mắt", "cao cấp", "độc đáo",
    "hợp khẩu vị", "rất tốt", "rất thích", "tận tâm", "đáng tin cậy", "đẳng cấp",
    "hấp dẫn", "an tâm", "không thể cưỡng lại", "thỏa mãn", "thúc đẩy",
    "cảm động", "phục vụ tốt", "làm hài lòng", "gây ấn tượng", "nổi trội",
    "sáng tạo", "quý báu", "phù hợp", "tận tâm",
    "hiếm có", "cải thiện", "hoà nhã", "chăm chỉ", "cẩn thận",
    "vui vẻ", "sáng sủa", "hào hứng", "đam mê", "vừa vặn", "đáng tiền", "nhiệt tình", "best", "good", "nghiện", "nhanh", "ngon nhất", "quá ngon", "quá tuyệt", "đúng vị", 
    "điểm cộng", "thức ăn ngon", "khá ngon", "niềm nở", "rất thích", "đặc biệt", "không bị", "tươi ngon", "thơm", "chất lượng", "rộng rãi", "tặng", "sạch sẽ", "món ngon", "ăn rất ngon", "giá rẻ",
    "thích nhất", "đồ ăn ngon", "phục vụ nhanh", "giá hợp", "đa dạng", "ngon giá", "phục vụ nhanh", "nhanh nhẹn", "thân thiện", "thơm", "ăn ngon", "cộng", "ủng_hộ quán", "ủng_hộ", "hấp_dẫn", "ấn_tượng", "thoải_mái",
    "quán ngon", "ủng_hộ", "khen", "dài_dài", "tin_tưởng"
]

# Tạo danh sách các từ khóa thuộc nhóm Negative
negative_words = [
    "kém", "tệ", "đau", "xấu", "không", "dở", "ức",
    "buồn", "rối", "thô", "lâu", "chán", "tối", "chán", "ít", "mờ", "mỏng",
    "lỏng lẻo", "khó", "cùi", "yếu",
    "kém chất lượng", "không thích", "không thú vị", "không ổn",
    "không hợp", "không đáng tin cậy", "không chuyên nghiệp",
    "không phản hồi", "không an toàn", "không phù hợp", "không thân thiện", "không linh hoạt", "không đáng giá",
    "không ấn tượng", "không tốt", "chậm", "khó khăn", "phức tạp",
    "khó hiểu", "khó chịu", "gây khó dễ", "rườm rà", "khó truy cập",
    "thất bại", "tồi tệ", "khó xử", "không thể chấp nhận", "tồi tệ", "không rõ ràng",
    "không chắc chắn", "rối rắm", "không tiện lợi", "không đáng tiền", "chưa đẹp", "không đẹp", "bad", "thất vọng", "không ngon", "không hợp",
    "hôi", "trộm cướp", "không_ngon", "không_thích", "không_ổn", "không_hợp", "lần cuối", "cuối cùng", "quá tệ", "quá dở", "quá mắc", "cau có", "không đáng", "chả đáng",
    "điểm trừ", "thức ăn tệ", "đồ ăn tệ", "đợi lâu", "nhạt nhẽo", "không thoải mái", "không đặc sắc", "tanh", "giá hơi mắc", "giá hơi đắt", "không chất lượng", "chê", "trừ",
    "giá hơi", "chậm", "chậm chạm", "lâu", "quá lâu", "nhạt", "chờ", "ăn hơi", "khủng khiếp", "đợi", "nhạt", "thất_vọng", "bực_mình"
]

# Tạo cột 'Positive Count' và 'Negative Count' dựa trên từ khóa positive_words và negative_words
data['Positive Count'] = data['Comment Tokenize'].apply(lambda x: sum(x.lower().count(word) for word in positive_words))
data['Negative Count'] = data['Comment Tokenize'].apply(lambda x: sum(x.lower().count(word) for word in negative_words))

# Lọc ra các bài đánh giá có 'Positive Count' và 'Negative Count' khác 0
positive_reviews = data[data['Positive Count'] > 0]
negative_reviews = data[data['Negative Count'] > 0]
data.to_csv('data_cleaned/data_analysis.csv', index=False)

# Tạo biểu đồ WordCloud cho các bài đánh giá tích cực
positive_text = ' '.join(positive_reviews['Comment Tokenize'])
wordcloud_positive = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(positive_text)

# Tạo biểu đồ WordCloud cho các bài đánh giá tiêu cực
negative_text = ' '.join(negative_reviews['Comment Tokenize'])
wordcloud_negative = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate(negative_text)

# Hiển thị biểu đồ WordCloud
plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.title('Word Cloud - Positive Reviews')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.title('Word Cloud - Negative Reviews')
plt.axis('off')

plt.tight_layout()
plt.show()

# Biểu đồ phân bố số lượng từ tích cực và tiêu cực
plt.figure(figsize=(10, 6))
sb.histplot(data['Positive Count'], bins=20, color='blue', kde=True, label='Positive Count')
sb.histplot(data['Negative Count'], bins=20, color='red', kde=True, label='Negative Count')
plt.title('Distribution of Positive and Negative Counts')
plt.xlabel('Count')
plt.legend()
plt.show()
