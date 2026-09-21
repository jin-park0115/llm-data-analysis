
import pandas as pd

import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.naive_bayes import MultinomialNB

DATA_PATH = "book_bestseller_clean.csv"

st.set_page_config(

    page_title="베스트셀러 텍스트 분석 앱",

    layout="wide",

)

@st.cache_data

def load_data():

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    required_columns = ["상품명", "분야"]

    missing_columns = [

        col for col in required_columns

        if col not in df.columns

    ]

    if missing_columns:

        raise ValueError(

            f"필수 컬럼이 없습니다: {missing_columns}"

        )

    df["상품명"] = (

        df["상품명"]

        .fillna("")

        .astype(str)

        .str.strip()

    )

    df["분야"] = (

        df["분야"]

        .fillna("미분류")

        .astype(str)

        .str.strip()

    )

    df = df[df["상품명"] != ""].reset_index(drop=True)

    return df

@st.cache_resource

def train_classifier():
    df = load_data()
    train_df = df[df["분야"] != "미분류"].copy()
    if train_df["분야"].nunique() < 2:
        raise ValueError(
            "서로 다른 분야가 2개 이상 필요합니다."
        )
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(train_df["상품명"])
    y = train_df["분야"]
    model = MultinomialNB()
    model.fit(X, y)
    return vectorizer, model
@st.cache_resource

def build_recommender():
    df = load_data()
    vectorizer = TfidfVectorizer()
    title_matrix = vectorizer.fit_transform(
        df["상품명"]
    )
    return vectorizer, title_matrix

def recommend_books(
    df,
    title_matrix,
    selected_index,
    top_n=5,
):

    similarities = cosine_similarity(
        title_matrix[selected_index],
        title_matrix,
    ).ravel()

    similarities[selected_index] = -1
    top_indices = similarities.argsort()[::-1][:top_n]
    display_columns = [
        col
        for col in ["상품명", "저자", "출판사", "분야"]
        if col in df.columns
    ]

    result = df.iloc[top_indices][display_columns].copy()
    result["유사도"] = similarities[top_indices].round(3)
    return result

try:

    df = load_data()
    classifier_vectorizer, classifier_model = train_classifier()
    _, title_matrix = build_recommender()

except (FileNotFoundError, ValueError) as error:

    st.error(str(error))
    st.stop()

st.title("교보문고 베스트셀러 텍스트 분석 앱")

st.write(

    "도서 제목을 이용한 분야 예측과 "
    "유사 도서 추천 기능을 실습합니다."

)

# 1. 분류

st.header("1. 도서 분야 예측")

user_title = st.text_input(
    "도서 제목을 입력하세요",
    placeholder="예: 처음 배우는 파이썬 데이터 분석",
)

if st.button("분야 예측"):
    clean_title = user_title.strip()

    if not clean_title:
        st.warning("도서 제목을 입력해 주세요.")
    else:
        title_vector = classifier_vectorizer.transform(
            [clean_title]
        )
        predicted_category = classifier_model.predict(
            title_vector
        )[0]
        st.success(f"예상 분야: {predicted_category}")
st.divider()

# 2. 추천

st.header("2. 비슷한 도서 추천")
def format_book(index):
    title = df.loc[index, "상품명"]
    if "저자" in df.columns:
        author = str(df.loc[index, "저자"])
        return f"{title} | {author}"
    return title

selected_index = st.selectbox(
    "기준 도서를 선택하세요",
    options=df.index.tolist(),
    format_func=format_book,
)

if st.button("비슷한 도서 5권 추천"):
    recommendations = recommend_books(
        df,
        title_matrix,
        selected_index,
        top_n=5,
    )

    st.dataframe(
        recommendations,
        width="stretch",
        hide_index=True,
    )

st.divider()

st.caption(
    "분류 결과는 제목의 텍스트 패턴을 이용한 예측이며 "
    "실제 서점의 공식 분류와 다를 수 있습니다."
)

st.caption(
    "추천 결과는 제목의 TF-IDF 코사인 유사도를 기반으로 하며 "
    "개별 사용자의 취향을 직접 반영하지 않습니다."
)