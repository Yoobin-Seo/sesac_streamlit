import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EDA 리포트 생성 앱", layout="wide")

st.title("📊 EDA 리포트 생성 Streamlit 앱")
st.write(
    "CSV 파일을 업로드하면 자동으로 탐색적 데이터 분석(EDA) 리포트를 생성합니다. "
    "6주차 강의에서 배운 AI 프로젝트 배포 실습용으로 제작된 앱입니다."
)

# ---------------------------------------------------------
# 1. 데이터 업로드
# ---------------------------------------------------------
uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

st.sidebar.markdown("---")
use_sample = st.sidebar.checkbox("샘플 데이터(tips) 사용하기", value=True if uploaded_file is None else False)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif use_sample:
    df = sns.load_dataset("tips")
else:
    st.info("왼쪽 사이드바에서 CSV 파일을 업로드하거나 샘플 데이터를 선택하세요.")
    st.stop()

# ---------------------------------------------------------
# 2. 데이터 개요
# ---------------------------------------------------------
st.header("1. 데이터 개요")
col1, col2, col3 = st.columns(3)
col1.metric("행 개수", df.shape[0])
col2.metric("열 개수", df.shape[1])
col3.metric("결측치 총합", int(df.isnull().sum().sum()))

st.dataframe(df.head())

# ---------------------------------------------------------
# 3. 결측치 리포트
# ---------------------------------------------------------
st.header("2. 결측치 확인")
missing_count = df.isnull().sum()
missing_pct = (missing_count / len(df) * 100).round(2)
missing_report = pd.DataFrame({"결측치 개수": missing_count, "결측치 비율(%)": missing_pct})
st.dataframe(missing_report.sort_values("결측치 개수", ascending=False))

# ---------------------------------------------------------
# 4. 기술 통계
# ---------------------------------------------------------
st.header("3. 기술 통계")
st.dataframe(df.describe().T)

# ---------------------------------------------------------
# 5. 상관관계 히트맵
# ---------------------------------------------------------
st.header("4. 상관관계 히트맵")
numeric_df = df.select_dtypes(include="number")
if numeric_df.shape[1] >= 2:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="Blues", ax=ax)
    st.pyplot(fig)
else:
    st.warning("상관관계를 계산할 수치형 컬럼이 2개 미만입니다.")

# ---------------------------------------------------------
# 6. 분포 시각화 (사용자 선택)
# ---------------------------------------------------------
st.header("5. 컬럼별 분포 시각화")
if len(numeric_df.columns) > 0:
    selected_col = st.selectbox("분포를 확인할 컬럼을 선택하세요", numeric_df.columns)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.histplot(df[selected_col], kde=True, ax=ax2)
    st.pyplot(fig2)
else:
    st.warning("시각화할 수치형 컬럼이 없습니다.")

st.markdown("---")
st.caption("AI for Future Workforce · Week 6: AI Project Deployment 실습용 앱")
