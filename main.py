
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

    # 전체 기간의 회귀 직선
    slope, intercept = np.polyfit(x, y, 1)
    correlation = np.corrcoef(x, y)[0, 1]

    # 100년당 기온 변화량
    rise_100_years = slope * 100

    # 최근 20년 데이터: 가장 최근 관측 연도를 기준으로 계산
    latest_year = int(yearly["연도"].max())
    recent_start_year = latest_year - 19

    recent = yearly[
        (yearly["연도"] >= recent_start_year)
        & (yearly["연도"] <= latest_year)
    ].copy()

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

    yearly["회귀 예측기온"] = slope * x + intercept

    # -----------------------------
    # 분석 기간 정보
    # -----------------------------
    start_year = int(yearly["연도"].min())
    end_year = int(yearly["연도"].max())
    data_count = len(yearly)

    col1, col2, col3 = st.columns(3)
    col1.metric("회귀 분석 연도 수", f"{data_count}개")
    col2.metric("시작 연도", f"{start_year}년")
    col3.metric("끝 연도", f"{end_year}년")

    st.subheader("🌡️ 100년당 기온 변화 비교")

    # 두 기울기를 나란히 크게 표시
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label=f"전체 기간 ({start_year}~{end_year}년)",
            value=f"{rise_100_years:+.2f} ℃ / 100년"
        )
        st.caption(
            "전체 분석 기간의 연평균기온으로 계산한 장기 추세"
        )

    with col2:
        if recent_rise_100_years is not None:
            st.metric(
                label=(
                    f"최근 20년 ({recent_start_year}"
                    f"~{latest_year}년)"
                ),
                value=f"{recent_rise_100_years:+.2f} ℃ / 100년"
            )
            st.caption(
                f"사용한 연도 수: {len(recent)}개"
            )
        else:
            st.warning("최근 기간의 데이터가 부족해 비교할 수 없습니다.")

    st.subheader("📊 상관계수 및 회귀 분석 정보")
    st.write(f"- 전체 기간 상관계수: **{correlation:.4f}**")

    if recent_correlation is not None:
        st.write(
            f"- 최근 20년 상관계수: **{recent_correlation:.4f}**"
        )

    st.write(
        f"- 전체 기간 회귀식: "
        f"y = {slope:.5f} × 연도 {intercept:+.3f}"
    )

    if recent_slope is not None:
        st.write(
            f"- 최근 20년 회귀식: "
            f"y = {recent_slope:.5f} × 연도 "
            f"{recent_intercept:+.3f}"
        )

    st.caption(
        "양수는 기온 상승 추세, 음수는 하락 추세를 뜻합니다. "
        "최근 20년은 데이터에 실제로 존재하는 가장 최근 연도를 "
        "끝 연도로 삼아 계산합니다."
    )
