import streamlit as st
import speech_recognition as sr
import openai
import pyodbc
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import os

# .env 파일의 환경 변수를 로드
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# 데이터베이스 연결 설정 (초기화 시 한 번만 연결하고 재사용)
@st.cache_resource
def get_database_connection():
    try:
        # 환경 변수에서 DATABASE_URL 가져오기
        db_connection_string = os.getenv("DATABASE_URL")
        if not db_connection_string:
            st.error("DATABASE_URL environment variable is not set.")
            return None
        
        # pyodbc.connect에 DATABASE_URL을 전달하여 연결 설정
        conn = pyodbc.connect(db_connection_string)
        return conn
    except pyodbc.Error as e:
        st.error(f"Database connection error: {e}")
        return None

conn = get_database_connection()

# RODB 데이터베이스 연결 및 TRMN_ID와 TRMN_NM 조회
@st.cache_resource
def query_edi_code():
    if not conn:
        st.error("No database connection available.")
        return []
    
    cursor = conn.cursor()
    try:
        query = """
        SELECT A.TRMN_ID, A.TRMN_NM
        FROM METRDCTMT A
        WHERE A.TRMN_ID LIKE 'D%'
        ORDER BY A.TRMN_ID
        """
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    except pyodbc.Error as e:
        st.error(f"Query execution error: {e}")
        return []

# RODB 데이터베이스 연결 및 상세 정보 조회
def query_detailed_info(trmn_id):
    if not conn:
        st.error("No database connection available.")
        return None
    
    cursor = conn.cursor()
    try:
        query = f"""
        SELECT A.TRMN_ID, A.TRMN_NM, A.MDCR_USE_YN, A.APLY_YMD, A.FNSH_YMD,
               B.SNOMED_MPNG_SNO, B.TRMN_TYPE_CD, B.SNOMED_TYPE_CD_1, B.SNOMED_TYPE_CD_2,
               B.VRSN_ID, B.SNOMED_CONP_ID, B.SNOMED_CONP_CTN, B.SNOMED_DSCR_ID, B.SNOMED_DSCR_CTN,
               B.APST_YMD AS S_APST_YMD, B.APFN_YMD AS S_APFN_YMD
        FROM METRDCTMT A, SNOMEDCTT B
        WHERE A.TRMN_ID = B.TRMN_ID AND A.TRMN_ID = '{trmn_id}'
        ORDER BY A.TRMN_ID, B.SNOMED_MPNG_SNO
        """
        cursor.execute(query)
        result = cursor.fetchone()
        return result
    except pyodbc.Error as e:
        st.error(f"Query execution error: {e}")
        return None

# Vector DB 학습 (TRMN_ID와 TRMN_NM만 임베딩)
@st.cache_resource
def train_vector_db(data):
    try:
        terms = [f"{row[0]} {row[1]}" for row in data]  # TRMN_ID + TRMN_NM 임베딩
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(terms)
        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X.toarray())
        return index, vectorizer
    except Exception as e:
        st.error(f"Error during vector DB training: {e}")
        return None, None

# Vector DB 검색
def search_vector_db(query, index, vectorizer):
    if not query:
        return []
    try:
        query_vec = vectorizer.transform([query]).toarray()
        _, I = index.search(query_vec, 5)
        return I
    except Exception as e:
        st.error(f"Error during vector DB search: {e}")
        return []

# 음성 인식을 통해 텍스트로 변환 (다국어 지원)
def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=1)
        r.pause_threshold = 3
        try:
            audio = r.listen(source)
            text = r.recognize_google(audio)  # 언어 자동 감지
            return text
        except sr.WaitTimeoutError:
            st.error("Speech recognition timed out. Please try again.")
        except sr.UnknownValueError:
            st.error("Google Speech Recognition could not understand the audio. Please speak more clearly.")
            st.button("Try again")
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service: {e}")
        return None

# LLM을 통해 의심되는 진단명 확인 (다국어 지원)
def get_diagnosis_suggestions(text):
    try:
        prompt = f"""
        You are an international hospital doctor. Please diagnose the symptoms that a foreign patient says. 
        Tell me the most appropriate diagnosis in order.
        Focus on specific and precise medical terms:

        Symptoms/Description: {text}
        """
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical expert providing precise diagnoses."},
                {"role": "user", "content": prompt},
            ]
        )
        suggestions = response['choices'][0]['message']['content'].splitlines()
        return [s.strip() for s in suggestions if s.strip()]
    except Exception as e:
        st.error(f"Error during AI doctor diagnosis suggestion: {e}")
        return []

# Streamlit 애플리케이션
def main():
    # CSS 스타일을 적용하여 제목과 테이블을 모두 가운데 정렬
    st.markdown(
        """
        <style>
        .center-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        table {
            margin-left: auto;
            margin-right: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 제목을 가운데로 정렬
    st.markdown("<div class='center-container'><h1>HUNIVERSE PHIS diagnosis📄</h1></div>", unsafe_allow_html=True)

    # Step 1: RODB에서 TRMN_ID와 TRMN_NM을 가져와 벡터 DB(=ME)에 임베딩
    with st.spinner("Embedding TRMN_ID and TRMN_NM into ME..."):
        data = query_edi_code()
        if data:
            index, vectorizer = train_vector_db(data)
            st.success("Ready")
        else:
            st.error("Failed to load data for ME.")

    # Step 2: 음성을 입력받아 텍스트로 변환
    if st.button("SPEAK SYMPTOM"):
        with st.spinner("Preparing for voice input..."):
            input_text = recognize_speech()
        if input_text:
            st.success("SYMPTOM input successfully processed.")

            # Step 3: LLM을 사용하여 진단명 확인
            with st.spinner("Identifying diagnosis using AI doctor..."):
                diagnosis_suggestions = get_diagnosis_suggestions(input_text)

            # Step 4: 벡터 DB에서 유사한 TRMN_ID와 TRMN_NM 검색
            with st.spinner("Searching for similar diagnoses within ME..."):
                if index and vectorizer:
                    results_indices = search_vector_db(input_text, index, vectorizer)
                    if results_indices.size > 0:
                        st.markdown("<div class='center-container'><p>Similar diagnoses found within ME:</p></div>", unsafe_allow_html=True)
                        for i in results_indices[0]:
                            row = data[i]
                            trmn_id = row[0]  # TRMN_ID를 가져옴

                            # Step 5: RODB에서 해당 TRMN_ID의 상세 정보를 가져옴
                            detailed_info = query_detailed_info(trmn_id)
                            if detailed_info:
                                # 데이터를 테이블로 구성
                                df = pd.DataFrame({
                                    "Column": [
                                        "TRMN ID", "Term Name",  "SNOMED Concept ID", 
                                        "SNOMED Concept Content", "SNOMED Description ID", 
                                        "SNOMED Description Content", "Apply Start Date", "Apply End Date"
                                    ],
                                    "Value": [
                                        detailed_info[0], detailed_info[1], detailed_info[10], detailed_info[11], 
                                        detailed_info[12], detailed_info[13], detailed_info[14], detailed_info[15]
                                    ]
                                })

                                # 테이블을 HTML로 변환하여 표시
                                html_table = df.to_html(index=False, escape=False, justify='center', border=0)
                                html_table = html_table.replace('<table border="1" class="dataframe">', '<table style="width:100%; table-layout: fixed; border-collapse: collapse; margin-left:auto; margin-right:auto;">')
                                html_table = html_table.replace('<th>', '<th style="background-color: #E0F7FA; padding: 8px; border: 1px solid black;">')
                                html_table = html_table.replace('<td>', '<td style="width: 300px; background-color: white; padding: 8px; border: 1px solid black; word-wrap: break-word;">')

                                # 컬럼 'TRMN ID'부터 'Apply End Date'에만 색상 적용
                                html_table = html_table.replace('<td>TRMN ID</td>', '<td style="background-color: #f0f0f0;">TRMN ID</td>')
                                html_table = html_table.replace('<td>Term Name</td>', '<td style="background-color: #f0f0f0;">Term Name</td>')
                                html_table = html_table.replace('<td>SNOMED Concept ID</td>', '<td style="background-color: #f0f0f0;">SNOMED Concept ID</td>')
                                html_table = html_table.replace('<td>SNOMED Concept Content</td>', '<td style="background-color: #f0f0f0;">SNOMED Concept Content</td>')
                                html_table = html_table.replace('<td>SNOMED Description ID</td>', '<td style="background-color: #f0f0f0;">SNOMED Description ID</td>')
                                html_table = html_table.replace('<td>SNOMED Description Content</td>', '<td style="background-color: #f0f0f0;">SNOMED Description Content</td>')
                                html_table = html_table.replace('<td>Apply Start Date</td>', '<td style="background-color: #f0f0f0;">Apply Start Date</td>')
                                html_table = html_table.replace('<td>Apply End Date</td>', '<td style="background-color: #f0f0f0;">Apply End Date</td>')

                                st.markdown(f"<div class='center-container'>{html_table}</div>", unsafe_allow_html=True)
                            else:
                                st.warning(f"No detailed information found for TRMN ID: {trmn_id}")
                    else:
                        st.warning("No matching diagnoses found.")
                else:
                    st.error("Vector DB is not ready. Cannot perform search.")

if __name__ == "__main__":
    main()


