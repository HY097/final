import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import io

st.set_page_config(page_title="서울 기온 추이 분석", layout="wide")

# ------------------------------------------------------------------
# 데이터 로드
# ------------------------------------------------------------------
@st.cache_data
def load_data_from_bytes(data: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
    df["날짜"] = df["날짜"].str.strip()
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연"] = df["날짜"].dt.year
    return df

uploaded = st.file_uploader("seoul_temperature.csv 파일을 선택하세요", type="csv")

if uploaded is not None:
    df = load_data_from_bytes(uploaded.read())
else:
    try:
        df = load_data_from_bytes(open("seoul_temperature.csv", "rb").read())
    except FileNotFoundError:
        st.error("seoul_temperature.csv 파일을 찾지 못했습니다.")
        st.info("① app.py와 같은 폴더에 seoul_temperature.csv를 두거나, ② 파일 업로드 창에서 파일을 선택하세요.")
        st.stop()

# 1) 연도별 평균기온 집계
annual = (
    df.groupby("연")["평균기온(℃)"]
    .mean()
    .reset_index()
    .rename(columns={"평균기온(℃)": "연평균기온"})
)

# 2) 5년 이동평균 (중앙 기준)
annual["이동평균5년"] = annual["연평균기온"].rolling(window=5, center=True).mean()

# ------------------------------------------------------------------
# 사이드바: 연도 범위 슬라이더
# ------------------------------------------------------------------
min_year, max_year = int(annual["연"].min()), int(annual["연"].max())
year_range = st.sidebar.slider("연도 범위 선택", min_year, max_year, (min_year, max_year))
start, end = year_range

# 선택 구간 필터 & 통계
mask = (annual["연"] >= start) & (annual["연"] <= end)
filtered = annual.loc[mask].copy()

sel_df = df[(df["연"] >= start) & (df["연"] <= end)]
stats = {
    "평균기온(℃)": sel_df["평균기온(℃)"].mean(),
    "최고기온(℃)": sel_df["최고기온(℃)"].max(),
    "최저기온(℃)": sel_df["최저기온(℃)"].min(),
}

st.subheader("선택 구간 통계")
st.write(f"**{start}년 ~ {end}년**")
for label, value in stats.items():
    st.metric(label=label, value=round(value, 1))

# 수정: columns 인자 제거 (최신 Streamlit은 columns 미지원)
st.dataframe(filtered, use_container_width=True)

# ------------------------------------------------------------------
# 메인 그래프: 연평균기온 + 5년 이동평균
# ------------------------------------------------------------------
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=filtered["연"],
        y=filtered["연평균기온"],
        mode="markers+lines",
        name="연평균기온",
        marker=dict(size=6),
    )
)
fig.add_trace(
    go.Scatter(
        x=filtered["연"],
        y=filtered["이동평균5년"],
        mode="lines",
        name="5년 이동평균",
        line=dict(color="red", width=2, dash="dash"),
    )
)
fig.update_layout(
    title="서울 연평균기온 및 5년 이동평균",
    xaxis_title="연도",
    yaxis_title="기온 (℃)",
    hovermode="x unified",
    legend=dict(x=0.02, y=0.98),
)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# 히트맵: 행=연도, 열=월, 셀값=월평균기온
# 색상 기준: 전체 월 평균기온 값의 중앙값
# ------------------------------------------------------------------
st.subheader("연대(년) x 월 평균기온 히트맵")

monthly = (
    df.groupby([df["날짜"].dt.year, df["날짜"].dt.month])["평균기온(℃)"]
    .mean()
    .reset_index()
)
pivot = monthly.pivot(index=0, columns=1, values="평균기온(℃)").reindex(columns=range(1, 13))

vals = pivot.values[~np.isnan(pivot.values)]
median_val = float(np.median(vals))
zmax = float(vals.max())
zmin = 2 * median_val - zmax  # 중앙값이 색상 중앙(흰색)에 오도록 설정

fig_heat = go.Figure(
    go.Heatmap(
        z=pivot.values,
        x=pivot.columns.astype(int),
        y=pivot.index.astype(int),
        colorscale="RdBu_r",
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(title="평균기온 (℃)", title_side="right"),
        hovertemplate="연도 %{y}년 · 월 %{x}월<br>평균기온 %{z:.1f}℃<extra></extra>",
    )
)
fig_heat.update_layout(
    title="연대·월별 평균기온 히트맵",
    xaxis_title="월",
    yaxis_title="연도",
)
st.plotly_chart(fig_heat, use_container_width=True)

st.caption(
    f"히트맵 색상 기준: 전체 월 평균기온 {len(vals)}개 값의 중앙값 = {median_val:.1f}℃"
)
