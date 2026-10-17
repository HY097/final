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
# 히트맵: 연대(년) x 월, 셀값=월평균기온
# 호버: 평균/최저/최고 표시, 컬러바 고정, 연도별 필터링 연동
st.subheader("연대(년) x 월 평균기온 히트맵")

df_heat = sel_df

# 연-월별 집계 (평균/최저/최고)
monthly_agg = (
    df_heat.assign(연=df_heat["날짜"].dt.year, 월=df_heat["날짜"].dt.month)
    .groupby(["연", "월"])
    .agg(월별평균=("평균기온(℃)", "mean"),
         월최저=("최저기온(℃)", "min"),
         월최고=("최고기온(℃)", "max"))
    .reset_index()
)

pivot_mean = monthly_agg.pivot(index="연", columns="월", values="월별평균").reindex(columns=range(1, 13))
pivot_min = monthly_agg.pivot(index="연", columns="월", values="월최저").reindex(index=pivot_mean.index, columns=pivot_mean.columns)
pivot_max = monthly_agg.pivot(index="연", columns="월", values="월최고").reindex(index=pivot_mean.index, columns=pivot_mean.columns)

# 호버 툴팁용 커스텀데이터: [최저기온, 최고기온] (shape: rows x cols x 2)
customdata = np.dstack((pivot_min.values, pivot_max.values))

# 컬러바 고정: 전체 연도 기준 최소/최대 (연도 범위 변경해도 색상 스케일이 동일)
monthly_all = (
    df.assign(연=df["날짜"].dt.year, 월=df["날짜"].dt.month)
    .groupby(["연", "월"])["평균기온(℃)"]
    .mean()
    .unstack("월")
)
global_zmin = float(np.nanmin(monthly_all.values))
global_zmax = float(np.nanmax(monthly_all.values))

fig_heat = go.Figure(
    go.Heatmap(
        z=pivot_mean.values,
        x=pivot_mean.columns.astype(int),
        y=pivot_mean.index.astype(int),
        colorscale="RdBu_r",
        zmin=global_zmin,
        zmax=global_zmax,
        customdata=customdata,
        hovertemplate=(
            "연도 %{y}년 · 월 %{x}월<br>"
            "평균기온 %{z:.1f}℃<br>"
            "최저기온 %{customdata[0]:.1f}℃<br>"
            "최고기온 %{customdata[1]:.1f}℃<extra></extra>"
        ),
    )
)
fig_heat.update_layout(
    title="연대·월별 평균기온 히트맵 (색상 기준: 전체 연도 공통 범위)",
    xaxis_title="월",
    yaxis_title="연도",
)
st.plotly_chart(fig_heat, use_container_width=True)

# 3) 월별 평균 라인 (선택 구간 기준, 히트맵과 동일 범위)
seasonal = pivot_mean.mean(axis=0)  # 월(1~12)별 평균

fig_season = go.Figure(go.Scatter(
    x=seasonal.index,
    y=seasonal.values,
    mode="lines+markers",
    name="월별 평균기온",
    line=dict(color="orange", width=2),
))
fig_season.update_layout(
    title="선택 구간 월별 평균기온",
    xaxis_title="월",
    yaxis_title="평균기온 (℃)",
    xaxis_tickvals=list(range(1, 13)),
    xaxis_ticktext=["1월","2월","3월","4월","5월","6월","7월","8월","9월","10월","11월","12월"],
)
st.plotly_chart(fig_season, use_container_width=True)

st.caption(
    f"히트맵 색상 기준: 전체 연도 공통 고정 범위 {global_zmin:.1f}℃ ~ {global_zmax:.1f}℃"
)
