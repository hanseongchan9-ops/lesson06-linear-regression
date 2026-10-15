
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 페이지 설정
# ==========================================
st.set_page_config(
    page_title="서울 기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 기온 예측기")
st.write(
    "서울의 과거 연평균기온을 분석하고, "
    "전체 기간과 최근 20년의 기온 변화 추세를 비교합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


# ==========================================
# 2. 데이터 불러오기 및 전처리
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 날짜와 평균기온을 올바른 자료형으로 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"], errors="coerce"
    )
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    # 날짜 또는 평균기온이 없는 자료 제거
    df = df.dropna(
        subset=["날짜", "평균기온"]
    ).copy()

    # 2025년까지의 자료만 사용
    df = df[df["날짜"].dt.year <= 2025].copy()

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 연도별 평균기온과 관측일 수 계산
    yearly = (
        df.groupby("연도")
        .agg(
            평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique")
        )
        .reset_index()
    )

    # 관측일이 300일 미만인 연도 제외
    yearly = yearly[
        yearly["관측일수"] >= 300
    ].copy()

    return yearly.sort_values(
        "연도"
    ).reset_index(drop=True)


# ==========================================
# 3. 회귀 분석
# ==========================================
try:
    yearly = load_data()

    if len(yearly) < 2:
        st.error("회귀 분석에 필요한 데이터가 부족합니다.")
        st.stop()

    # 전체 기간 데이터
    x = yearly["연도"].to_numpy(dtype=float)
    y = yearly["평균기온"].to_numpy(dtype=float)

    # 전체 기간의 회귀 직선: y = slope * x + intercept
    slope, intercept = np.polyfit(x, y, 1)

    # 전체 기간의 상관계수
    correlation = np.corrcoef(x, y)[0, 1]

    # 100년당 기온 변화량
    rise_100_years = slope * 100

    # 분석 기간 정보
    start_year = int(yearly["연도"].min())
    end_year = int(yearly["연도"].max())
    data_count = len(yearly)

    # 최근 관측 연도를 기준으로 20년 범위 설정
    recent_start_year = end_year - 19

    recent = yearly[
        (yearly["연도"] >= recent_start_year)
        & (yearly["연도"] <= end_year)
    ].copy()

    # 최근 20년 회귀 분석
    if len(recent) >= 2:
        recent_x = recent["연도"].to_numpy(dtype=float)
        recent_y = recent["평균기온"].to_numpy(dtype=float)

        recent_slope, recent_intercept = np.polyfit(
            recent_x, recent_y, 1
        )

        recent_rise_100_years = recent_slope * 100
        recent_correlation = np.corrcoef(
            recent_x, recent_y
        )[0, 1]
    else:
        recent_slope = None
        recent_intercept = None
        recent_rise_100_years = None
        recent_correlation = None

    # 전체 기간의 회귀 예측값
    yearly["회귀 예측기온"] = slope * x + intercept

    # ==========================================
    # 4. 분석 기간 표시
    # ==========================================
    st.subheader("📊 분석 데이터")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "회귀 분석에 사용한 연도",
        f"{data_count}개"
    )
    col2.metric(
        "시작 연도",
        f"{start_year}년"
    )
    col3.metric(
        "끝 연도",
        f"{end_year}년"
    )

    st.caption(
        "2025년 이후의 자료와 관측일이 300일 미만인 연도는 제외했습니다."
    )

    # ==========================================
    # 5. 100년당 기온 변화량 비교
    # ==========================================
    st.subheader("🌡️ 100년에 기온이 얼마나 변할까?")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label=f"전체 기간 ({start_year}~{end_year}년)",
            value=f"{rise_100_years:+.2f} ℃ / 100년",
            help="전체 분석 기간의 연평균기온으로 계산한 회귀 직선의 기울기"
        )

    with col2:
        if recent_rise_100_years is not None:
            st.metric(
                label=f"최근 20년 ({recent_start_year}~{end_year}년)",
                value=f"{recent_rise_100_years:+.2f} ℃ / 100년",
                help="최근 20년 범위에 포함된 유효한 연평균기온으로 계산한 기울기"
            )
            st.caption(
                f"최근 기간의 실제 사용 연도 수: {len(recent)}개"
            )
        else:
            st.warning(
                "최근 기간에 유효한 연도가 2개 미만이라 "
                "기울기를 계산할 수 없습니다."
            )

    st.info(
        "양수(+)는 기온 상승 추세, 음수(-)는 기온 하락 추세를 의미합니다. "
        "100년당 변화량은 회귀 직선의 기울기에 100을 곱해 계산합니다."
    )

    # ==========================================
    # 6. 상관계수와 회귀식
    # ==========================================
    with st.expander("상관계수와 회귀식 자세히 보기"):
        st.write(f"**전체 기간 상관계수:** {correlation:.4f}")

        st.write(
            f"**전체 기간 회귀식:** "
            f"y = {slope:.5f} × 연도 {intercept:+.3f}"
        )

        if recent_slope is not None:
            st.write(
                f"**최근 20년 상관계수:** "
                f"{recent_correlation:.4f}"
            )
            st.write(
                f"**최근 20년 회귀식:** "
                f"y = {recent_slope:.5f} × 연도 "
                f"{recent_intercept:+.3f}"
            )

    # ==========================================
    # 7. Plotly 산점도 및 회귀 직선
    # ==========================================
    st.subheader("📈 연도별 평균기온과 회귀 직선")

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

    # 전체 기간 회귀 직선: 2100년까지 연장
    full_line_x = np.array([start_year, 2100], dtype=float)
    full_line_y = slope * full_line_x + intercept

    fig.add_trace(
        go.Scatter(
            x=full_line_x,
            y=full_line_y,
            mode="lines",
            name="전체 기간 회귀 직선",
            line=dict(width=3)
        )
    )

    # 최근 20년 회귀 직선
    if recent_slope is not None:
        recent_line_x = np.array(
            [
                int(recent["연도"].min()),
                end_year
            ],
            dtype=float
        )

        recent_line_y = (
            recent_slope * recent_line_x
            + recent_intercept
        )

        fig.add_trace(
            go.Scatter(
                x=recent_line_x,
                y=recent_line_y,
                mode="lines",
                name="최근 20년 회귀 직선",
                line=dict(width=3, dash="dash")
            )
        )

    fig.update_layout(
        xaxis_title="연도",
        yaxis_title="연평균기온 (℃)",
        template="plotly_white",
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==========================================
    # 8. 연도별 예상 기온
    # ==========================================
    st.subheader("🔮 원하는 연도의 예상 기온")

    selected_year = st.slider(
        "예측할 연도를 선택하세요",
        min_value=1900,
        max_value=2100,
        value=2026,
        step=1
    )

    # 전체 기간의 회귀 직선을 기준으로 예상 기온 계산
    predicted_temp = slope * selected_year + intercept

    st.metric(
        label=f"{selected_year}년 예상 연평균기온",
        value=f"{predicted_temp:.2f} ℃"
    )

    st.caption(
        "예상 기온은 전체 분석 기간의 선형 회귀식으로 계산한 추정치입니다. "
        "실제 기상 예보가 아니며, 먼 미래에는 오차가 클 수 있습니다."
    )

    # ==========================================
    # 9. 연도별 데이터 표
    # ==========================================
    with st.expander("연도별 분석 데이터 확인"):
        display_data = yearly.copy()

        display_data["평균기온"] = (
            display_data["평균기온"].round(2)
        )
        display_data["회귀 예측기온"] = (
            display_data["회귀 예측기온"].round(2)
        )

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )

except Exception as e:
    st.error(
        "데이터를 불러오거나 분석하는 중 오류가 발생했습니다. "
        "인터넷 연결과 CSV 주소를 확인해 주세요."
    )
    st.exception(e)
