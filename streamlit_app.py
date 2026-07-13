import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

model_name = "Gradient Boosting"
st.set_page_config(page_title="Personality Prediction", layout="wide")
st.title("🧠 외향형/내향형 예측 Streamlit 앱")
st.write(f"{model_name}을 이용한 Personality Classification")


def evaluate(df, model):

    # target 설정
    y = df["Personality_Introvert"]
    X = df.drop(columns=["Personality_Introvert"])

    # train, test 나누기
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 학습
    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict(X_test)

    # 정확도
    acu = accuracy_score(y_test, y_pred)

    return acu


tab1, tab2, tab3 = st.tabs(["📊 데이터 전처리", "📌 모델 정보", "🤖 모델 결과"])

with tab1:
    # 데이터 불러오기
    st.subheader("1. 원본 데이터")
    df = pd.read_csv("train.csv")
    col1, col2, col3 = st.columns(3)
    col1.metric("행 개수", df.shape[0])
    col2.metric("열 개수", df.shape[1])
    col3.metric("결측치 개수 총합", int(df.isnull().sum().sum()))
    with st.expander("원본 데이터 보기"):
        st.dataframe(df)
    st.divider()

    # 데이터 ignore
    st.subheader("2. 데이터 선별")
    df = df.drop(columns=["id"])
    st.info("""
        관련도 낮은 칼럼 무시 : id

        Target 설정: Personality
        """)
    st.divider()

    # 결측치 현황
    st.subheader("3. 결측치 처리")

    st.write("#### 결측치 현황")
    # 컬럼별 결측치 개수
    missing_count = df.isnull().sum()

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.barplot(x=missing_count.index, y=missing_count.values, ax=ax)

    ax.set_title("Missing count of Features")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing Count")
    plt.xticks(rotation=45, ha="right")

    st.pyplot(fig)

    # 결측칼럼 수 연산
    st.write("#### 새로운 칼럼 추가(인당 결측 데이터 개수)")
    df["n_missing"] = df.isnull().sum(axis=1)
    st.dataframe(df[["n_missing", "Personality"]].head())

    # 결측치 Average/Most frequent
    # 숫자형 컬럼
    st.write("#### 결측치 처리")
    st.write("-> 숫자형 컬럼은 평균값, 범주형 칼럼은 최빈값으로 대체")
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    # 범주형 컬럼
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    st.write(f"#### 처리 후 결측치 개수 총합 : {int(df.isnull().sum().sum())}")
    st.divider()

    # continuize 범주형만 원핫인코딩
    st.subheader("4. Continuize")
    st.write("범주형 데이터를 One-Hot Encoding 수행")
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    st.divider()
    st.subheader("전처리 후 데이터")
    col1, col2 = st.columns(2)
    col1.metric("컬럼 수", df.shape[1])
    col2.metric("결측치", int(df.isnull().sum().sum()))
    with st.expander("데이터 보기"):
        st.dataframe(df)


with tab2:
    param_df = pd.DataFrame(
        {
            "Parameter": [
                "model_name",
                "n_estimators",
                "learning_rate",
                "max_depth",
                "min_samples_split",
            ],
            "Value": [model_name, 100, 0.1, 5, 2],
        }
    )

    st.table(param_df)
with tab3:
    # 모델 적용
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.100,
        max_depth=5,
        min_samples_split=2,
        random_state=42,
    )
    st.subheader("Accuracy")
    st.metric("", f"{evaluate(df,model):.3f}")
    st.divider()
    st.subheader("칼럼 별 중요도")
    importance = pd.DataFrame(
        {
            "Feature": df.drop(columns=["Personality_Introvert"]).columns,
            "Importance": model.feature_importances_,
        }
    )
    importance = importance.sort_values("Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=importance, x="Importance", y="Feature", ax=ax)

    st.pyplot(fig)
    st.dataframe(importance.sort_values("Importance", ascending=False))
