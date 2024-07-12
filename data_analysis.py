import pandas as pd
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

# Lấy cột 'Comment Tokenize<tab<tab>>
text = data['Comment Tokenize']

# Thiết lập và fit vectorizer
tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    min_df=0.02,
    max_df=0.9
)
text_transformed = tfidf.fit_transform(text)

# Tạo DataFrame từ text_transformed
df_text = pd.DataFrame(text_transformed.toarray(), columns=tfidf.get_feature_names_out())

# Tạo danh sách các từ khóa tích cực và tiêu cực
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

# Xác định đánh giá là Positive, Negative hoặc Neutral
def classify_sentiment(row):
    if row['Positive Count'] > row['Negative Count']:
        return 'Positive'
    elif row['Negative Count'] > row['Positive Count']:
        return 'Negative'
    else:
        return 'Neutral'

# Tạo cột 'Label' dựa trên hàm classify_sentiment
data['Label'] = data.apply(classify_sentiment, axis=1)

# Lưu dữ liệu đã được xử lý vào file CSV
data.to_csv('data_cleaned/data_analysis.csv', index=False)
