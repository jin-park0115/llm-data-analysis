import pandas as pd
import numpy as np
import streamlit as st
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. 페이지 기본 설정 및 전처리/모델 로딩 (캐싱 활용) ---
st.set_page_config(page_title="도서 분류 및 추천 앱", layout="wide")

@st.cache_resource
def load_kiwi():
    return Kiwi()

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("books_improved.csv", encoding="utf-8-sig")
    kiwi = load_kiwi()
    stopwords = {"에디션"}
    
    # 텍스트 정제 함수 정의
    def preprocess_text(text):
        if not isinstance(text, str) or not text.strip():
            return ""
        # Kiwi 토큰화 (NNG, NNP, SL 추출)
        tokens = [
            token.form for token in kiwi.tokenize(text) 
            if token.tag in ['NNG', 'NNP', 'SL']
        ]
        # 불필요한 단어 및 필터링 적용 (길이 <= 1, 숫자만 있는 경우, 불용어)
        cleaned = [
            token for token in tokens 
            if len(token) > 1 and not token.isdigit() and token not in stopwords
        ]
        return " ".join(cleaned)

    # 전처리 컬럼 생성
    df["상품명_정제"] = df["상품명"].apply(preprocess_text)
    return df

@st.cache_resource
def train_classification_model(df):
    # 분류 모델 학습 (전체 정제 데이터 기반)
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(df["상품명_정제"])
    
    model = MultinomialNB()
    model.fit(X_vec, df["분야"])
    
    return vectorizer, model

# 로딩 수행
kiwi = load_kiwi()
df = load_and_preprocess_data()
classifier_vectorizer, classifier_model = train_classification_model(df)

# 불용어 정제 단일 함수 (사용자 입력용)
def preprocess_input(text):
    stopwords = {"에디션"}
    tokens = [
        token.form for token in kiwi.tokenize(text) 
        if token.tag in ['NNG', 'NNP', 'SL']
    ]
    cleaned = [
        token for token in tokens 
        if len(token) > 1 and not token.isdigit() and token not in stopwords
    ]
    return " ".join(cleaned)


# --- 2. 앱 타이틀 및 사이드바 메뉴 ---
st.title("📚 도서 분야 예측 및 연관 도서 추천 시스템")
menu = st.sidebar.radio("메뉴 선택", ["1. 도서 분야 예측", "2. 비슷한 도서 추천"])

# --- 메뉴 1. 도서 분야 예측 ---
if menu == "1. 도서 분야 예측":
    st.header("🔍 도서 분야 예측")
    st.caption("새로운 도서 제목을 입력하면 형태소 분석과 정제를 거쳐 해당 분야를 예측합니다.")
    
    input_title = st.text_input("도서 제목을 입력하세요:", placeholder="예: 처음 배우는 파이썬 데이터 분석")
    
    if st.button("분야 예측하기"):
        if not input_title.strip():
            st.warning("도서 제목을 입력해 주세요.")
        else:
            # 1. 입력 제목 전처리 (형태소 분석 + 필터링)
            cleaned_title = preprocess_input(input_title)
            
            st.write("**전처리 결과 (키워드 추출):**", f"`{cleaned_title}`" if cleaned_title else "(추출된 키워드가 없습니다)")
            
            if not cleaned_title:
                st.error("분석 가능한 유효 키워드가 없습니다. 다른 제목을 입력해 주세요.")
            else:
                # 2. 학습된 TF-IDF로 transform만 진행 (fit_transform 금지)
                input_vec = classifier_vectorizer.transform([cleaned_title])
                
                # 3. Naive Bayes 모델로 분야 예측
                predicted_category = classifier_model.predict(input_vec)[0]
                
                # 4. 결과 출력
                st.success(f"예상 분야: **[{predicted_category}]**")

# --- 메뉴 2. 비슷한 도서 추천 ---
elif menu == "2. 비슷한 도서 추천":
    st.header("📖 비슷한 도서 추천")
    st.caption("선택한 도서와 같은 분야 내에서 양수 코사인 유사도(Similarity > 0)를 가진 도서를 추천합니다.")
    
    # 도서 선택 드롭다운
    book_list = df["상품명"].tolist()
    selected_book = st.selectbox("추천 기준 도서를 선택하세요:", book_list)
    
    if st.button("도서 추천받기"):
        # 1. 선택한 도서 정보 추출
        selected_idx = df[df["상품명"] == selected_book].index[0]
        selected_category = df.loc[selected_idx, "분야"]
        selected_text = df.loc[selected_idx, "상품명_정제"]
        
        st.write(f"**기준 도서:** `{selected_book}` | **분야:** `{selected_category}`")
        
        # 2. 동일 분야 후보 선별 (자기 자신 제외)
        candidate_df = df[(df["분야"] == selected_category) & (df.index != selected_idx)].copy().reset_index(drop=True)
        
        if candidate_df.empty:
            st.info("현재 기준으로 같은 분야 내에 추천할 다른 도서가 없습니다.")
        else:
            # 3. 추천 후보들과 TF-IDF 및 코사인 유사도 계산
            rec_vectorizer = TfidfVectorizer()
            all_texts = [selected_text] + candidate_df["상품명_정제"].tolist()
            tfidf_matrix = rec_vectorizer.fit_transform(all_texts)
            
            sim_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            candidate_df["similarity"] = sim_scores
            
            # 4. 유사도 > 0 필터링
            valid_candidates = candidate_df[candidate_df["similarity"] > 0].copy()
            
            # 5. 결과 출력 (Top 5 또는 추천 없음 안내)
            if valid_candidates.empty:
                st.info("현재 기준으로 유사도가 있는 추천 도서를 찾지 못했습니다.")
            else:
                recommendations = valid_candidates.sort_values(by="similarity", ascending=False).head(5)
                st.subheader("Top 5 추천 목록")
                st.dataframe(
                    recommendations[["상품명", "분야", "similarity"]].rename(
                        columns={"상품명": "추천 도서명", "분야": "분야", "similarity": "유사도"}
                    ),
                    use_container_width=True
                )