
import streamlit as st
import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import random

nltk.download('movie_reviews')

@st.cache_data
def load_data():
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    random.shuffle(documents)
    texts = [" ".join(words) for words, label in documents]
    labels = [1 if label == "pos" else 0 for words, label in documents]
    return texts, labels

@st.cache_resource
def train_model(texts, labels):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_vec = vectorizer.fit_transform(texts)
    model = LogisticRegression()
    model.fit(X_vec, labels)
    return model, vectorizer

def run_prediction(text, model, vectorizer):
    user_vec = vectorizer.transform([text])
    prediction = model.predict(user_vec)[0]
    proba = model.predict_proba(user_vec)[0]
    label = "TÍCH CỰC 😄" if prediction == 1 else "TIÊU CỰC 😠"
    st.subheader(f"Kết quả: {label}")
    st.write(f"Xác suất tích cực: `{proba[1]:.2f}`, tiêu cực: `{proba[0]:.2f}`")

def main():
    st.title("🎭 Phân tích cảm xúc bình luận phim (IMDb)")
    st.write("Nhập một đoạn văn/bình luận và hệ thống sẽ phân loại là tích cực hoặc tiêu cực.")

    texts, labels = load_data()
    model, vectorizer = train_model(texts, labels)

    user_input = st.text_area("✍️ Nhập nội dung bình luận:")

    if st.button("Phân tích"):
        if user_input.strip() == "":
            st.warning("Vui lòng nhập nội dung để phân tích.")
        else:
            run_prediction(user_input, model, vectorizer)

if __name__ == "__main__":
    main()
