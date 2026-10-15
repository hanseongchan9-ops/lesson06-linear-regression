
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 기온 예측기")
st.write(
    "서울의 과거 연평균기온을 분석하고 "
    "선형 회귀를 이용해 선택한 연도의 예상 기온을 확인합니다."
)

# -----------------------------
# 데이터 불러오기
# -----------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 날짜를 날짜 형식으로 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 기온을 숫자로 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    # 유효한 날짜와 평균기온만 사용
    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 수업 기준인 2025년까지의 자료만 사용
    df = df[df["날짜"].dt.year <= 2025].copy()

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 연도별 관측일 수와 평균기온 계산
    yearly = (
        df.groupby("연도")
        .agg(
            평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 관측일이 300일 미만인 연도 제외
    yearly = yearly[yearly["관측일수"] >= 300].copy()

    return yearly.sort_values("연도").reset_index(drop=True)


# -----------------------------
# 데이터 처리 및 회귀 분석
# -----------------------------
try:
    yearly = load_data()

    if len(yearly) < 2:
        st.error("회귀 분석을 수행하기 위한 데이터가 부족합니다.")
        st.stop()

    x = yearly["연도"].to_numpy(dtype=float)
    y = yearly["평균기온"].to_numpy(dtype=float)

    # 최소제곱법으로 회귀 직선 계산
    slope, intercept = np.polyfit(x, y, 1)

    # 상관계수
    correlation = np.corrcoef(x, y)[0, 1]

    # 회귀 직선의 예측값
    yearly["회귀 예측기온"] = slope * x + intercept

    # -----------------------------
    # 분석 결과
    # -----------------------------
    start_year = int(yearly["연도"].min())
    end_year = int(yearly["연도"].max())
    data_count = len(yearly)

    col1, col2, col3 = st.columns(3)

    col1.metric("회귀 분석에 사용한 연도 수", f"{data_count}개")
    col2.metric("시작 연도", f"{start_year}년")
    col3.metric("끝 연도", f"{end_year}년")

    st.subheader("📊 분석 정보")
    st.write(f"- 상관계수: **{correlation:.4f}**")
    st.write(
        f"- 회귀 직선: **y = {slope:.5f} × 연도 "
        f"{intercept:+.3f}**"
    )
    st.caption(
        "2025년 이후의 관측 자료와 관측일이 300일 미만인 연도는 "
        "회귀 분석에서 제외했습니다."
    )

    # -----------------------------
    # 연도별 산점도와 회귀 직선
    # -----------------------------
    st.subheader("서울 연평균기온과 회귀 직선")

    fig = go.Figure()

    # 실제 연평균기온 산점도
    fig.add_trace(
        go.Scatter(
            x=yearly["연도"],
            y=yearly["평균기온"],
            mode="markers",
            name="실제 연평균기온",
            marker=dict(size=7),
            customdata=yearly[["관측일수"]],
            hovertemplate=(
                "연도: %{x}년<br>"
                "평균기온: %{y:.2f}℃<br>"
                "관측일수: %{customdata[0]}일"
                "<extra></extra>"
            )
        )
    )

    # 회귀 직선
    line_years = np.array([start_year, 2100])
    line_temps = slope * line_years + intercept

    fig.add_trace(
        go.Scatter(
            x=line_years,
            y=line_temps,
            mode="lines",
            name="선형 회귀 직선",
            line=dict(width=3)
        )
    )

    fig.update_layout(
        xaxis_title="연도",
        yaxis_title="연평균기온 (℃)",
        hovermode="closest",
        legend=dict(orientation="h", y=1.08),
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # 연도 선택 및 기온 예측
    # -----------------------------
    st.subheader("🔮 원하는 연도의 예상 기온")

    selected_year = st.slider(
        "예측할 연도를 선택하세요",
        min_value=1900,
        max_value=2100,
        value=2026,
        step=1
    )

    predicted_temp = slope * selected_year + intercept

    st.metric(
        label=f"{selected_year}년 예상 연평균기온",
        value=f"{predicted_temp:.2f} ℃"
    )

    st.caption(
        "예상 기온은 과거 연평균기온에 직선 회귀 모델을 적용한 "
        "추정치이며, 실제 기상 예보가 아닙니다."
    )

    # -----------------------------
    # 연도별 데이터 표
    # -----------------------------
    with st.expander("연도별 분석 데이터 확인"):
        display_data = yearly.copy()
        display_data["평균기온"] = display_data["평균기온"].round(2)
        display_data["회귀 예측기온"] = (
            display_data["회귀 예측기온"].round(2)
        )
        st.dataframe(display_data, use_container_width=True)

except Exception as e:
    st.error(
        "데이터를 불러오거나 분석하는 중 오류가 발생했습니다. "
        "인터넷 연결과 CSV 주소를 확인해 주세요."
    )
    st.exception(e)
