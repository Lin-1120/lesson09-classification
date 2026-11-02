
import pandas as pd
import plotly.express as px
import streamlit as st

# ==========================================
# 기본 설정
# ==========================================

DATA_URL = "https://raw.githubusercontent.com/Lin-1120/data/refs/heads/main/stroke.csv"

st.title("🔎 데이터 탐색")
st.caption("뇌졸중 데이터의 분포와 변수 사이의 관계를 살펴봅니다.")

# ==========================================
# 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )
    return df


try:
    df = load_data()

except Exception as error:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(error))
    st.stop()


# ==========================================
# 데이터 전처리
# ==========================================

required_columns = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
    "bmi",
    "smoking_status",
    "stroke"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        "필요한 열이 없습니다: "
        + ", ".join(missing_columns)
    )
    st.stop()

# 분석에 필요한 숫자형 열 변환
for col in [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
    "stroke",
    "bmi"
]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# 뇌졸중 여부가 확인되는 데이터만 분석
analysis_df = df[df["stroke"].isin([0, 1])].copy()

if analysis_df.empty:
    st.error("분석할 수 있는 데이터가 없습니다.")
    st.stop()

analysis_df["stroke_label"] = analysis_df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})

# ==========================================
# 1. 나이와 평균 혈당 분포
# ==========================================

st.header("1. 나이와 평균 혈당의 분포")

st.write(
    """
    히스토그램은 데이터가 어느 구간에 많이 모여 있는지
    확인할 수 있는 그래프입니다.
    """
)

col1, col2 = st.columns(2)

with col1:

    st.subheader("나이 분포")

    age_data = analysis_df.dropna(
        subset=["age"]
    )

    fig_age = px.histogram(
        age_data,
        x="age",
        nbins=30,
        title="나이별 사람 수",
        labels={
            "age": "나이",
            "count": "사람 수"
        },
        color_discrete_sequence=["#5B9BD5"]
    )

    fig_age.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수",
        bargap=0.08
    )

    st.plotly_chart(
        fig_age,
        use_container_width=True
    )

with col2:

    st.subheader("평균 혈당 분포")

    glucose_data = analysis_df.dropna(
        subset=["avg_glucose_level"]
    )

    fig_glucose = px.histogram(
        glucose_data,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당별 사람 수",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수"
        },
        color_discrete_sequence=["#F4A261"]
    )

    fig_glucose.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수",
        bargap=0.08
    )

    st.plotly_chart(
        fig_glucose,
        use_container_width=True
    )


# ==========================================
# 2. 뇌졸중 여부에 따른 나이와 혈당 비교
# ==========================================

st.header("2. 뇌졸중 여부에 따른 비교")

st.write(
    """
    뇌졸중을 겪은 사람과 겪지 않은 사람의
    나이와 평균 혈당 분포를 상자그림으로 비교합니다.
    """
)

col1, col2 = st.columns(2)

with col1:

    st.subheader("나이 비교")

    age_box_data = analysis_df.dropna(
        subset=["age"]
    )

    fig_age_box = px.box(
        age_box_data,
        x="stroke_label",
        y="age",
        color="stroke_label",
        category_orders={
            "stroke_label": [
                "뇌졸중 없음",
                "뇌졸중 있음"
            ]
        },
        title="뇌졸중 여부에 따른 나이",
        labels={
            "stroke_label": "뇌졸중 여부",
            "age": "나이"
        },
        color_discrete_map={
            "뇌졸중 없음": "#5B9BD5",
            "뇌졸중 있음": "#E76F51"
        }
    )

    st.plotly_chart(
        fig_age_box,
        use_container_width=True
    )

with col2:

    st.subheader("평균 혈당 비교")

    glucose_box_data = analysis_df.dropna(
        subset=["avg_glucose_level"]
    )

    fig_glucose_box = px.box(
        glucose_box_data,
        x="stroke_label",
        y="avg_glucose_level",
        color="stroke_label",
        category_orders={
            "stroke_label": [
                "뇌졸중 없음",
                "뇌졸중 있음"
            ]
        },
        title="뇌졸중 여부에 따른 평균 혈당",
        labels={
            "stroke_label": "뇌졸중 여부",
            "avg_glucose_level": "평균 혈당"
        },
        color_discrete_map={
            "뇌졸중 없음": "#5B9BD5",
            "뇌졸중 있음": "#E76F51"
        }
    )

    st.plotly_chart(
        fig_glucose_box,
        use_container_width=True
    )


# ==========================================
# 두 그룹의 평균값 비교 표
# ==========================================

st.subheader("두 그룹의 평균값 비교")

mean_table = (
    analysis_df
    .groupby("stroke_label")[
        ["age", "avg_glucose_level"]
    ]
    .mean()
    .reindex([
        "뇌졸중 없음",
        "뇌졸중 있음"
    ])
    .rename(columns={
        "age": "평균 나이",
        "avg_glucose_level": "평균 혈당"
    })
    .round(2)
)

st.dataframe(
    mean_table,
    use_container_width=True
)


# ==========================================
# 3. 고혈압과 심장병에 따른 뇌졸중 비율
# ==========================================

st.header("3. 고혈압과 심장병에 따른 뇌졸중 비율")

st.write(
    """
    고혈압과 심장병이 있는 그룹과 없는 그룹의
    뇌졸중 비율을 비교합니다.
    """
)


def make_rate_chart(data, column, title):

    temp = data.dropna(
        subset=[column, "stroke"]
    ).copy()

    # 0과 1만 분석
    temp = temp[temp[column].isin([0, 1])]

    temp["group"] = temp[column].map({
        0: "없음",
        1: "있음"
    })

    result = (
        temp.groupby("group")
        .agg(
            전체_사람수=("stroke", "count"),
            뇌졸중_사람수=("stroke", "sum")
        )
        .reset_index()
    )

    result["뇌졸중_비율"] = (
        result["뇌졸중_사람수"]
        / result["전체_사람수"]
        * 100
    )

    result["뇌졸중_비율"] = result["뇌졸중_비율"].round(2)

    result["group"] = pd.Categorical(
        result["group"],
        categories=["없음", "있음"],
        ordered=True
    )

    result = result.sort_values("group")

    fig = px.bar(
        result,
        x="group",
        y="뇌졸중_비율",
        text="뇌졸중_비율",
        title=title,
        labels={
            "group": "질환 여부",
            "뇌졸중_비율": "뇌졸중 비율 (%)"
        },
        color="group",
        color_discrete_map={
            "없음": "#5B9BD5",
            "있음": "#E76F51"
        },
        hover_data={
            "전체_사람수": True,
            "뇌졸중_사람수": True,
            "뇌졸중_비율": ":.2f"
        }
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="뇌졸중 비율 (%)",
        xaxis_title="질환 여부",
        yaxis_range=[
            0,
            max(5, result["뇌졸중_비율"].max() * 1.2)
        ],
        showlegend=False
    )

    return fig, result


col1, col2 = st.columns(2)

with col1:

    st.subheader("고혈압 여부")

    hypertension_fig, hypertension_result = make_rate_chart(
        analysis_df,
        "hypertension",
        "고혈압 여부에 따른 뇌졸중 비율"
    )

    st.plotly_chart(
        hypertension_fig,
        use_container_width=True
    )

with col2:

    st.subheader("심장병 여부")

    heart_fig, heart_result = make_rate_chart(
        analysis_df,
        "heart_disease",
        "심장병 여부에 따른 뇌졸중 비율"
    )

    st.plotly_chart(
        heart_fig,
        use_container_width=True
    )


# ==========================================
# 4. BMI 결측치와 뇌졸중 비율
# ==========================================

st.header("4. BMI가 비어 있는 사람들의 특징")

st.write(
    """
    BMI가 비어 있는 사람들만 따로 모아
    그들의 뇌졸중 비율을 전체 데이터의 비율과 비교합니다.
    """
)

# BMI 결측 여부와 관계없이 stroke가 확인되는 사람 기준
missing_bmi = analysis_df[
    analysis_df["bmi"].isna()
]

total_count = len(analysis_df)
missing_count = len(missing_bmi)

total_stroke_count = int(
    (analysis_df["stroke"] == 1).sum()
)

missing_stroke_count = int(
    (missing_bmi["stroke"] == 1).sum()
)

total_rate = (
    total_stroke_count / total_count * 100
    if total_count > 0 else 0
)

missing_rate = (
    missing_stroke_count / missing_count * 100
    if missing_count > 0 else 0
)

bmi_comparison = pd.DataFrame({
    "구분": [
        "전체 사람",
        "BMI가 비어 있는 사람"
    ],
    "사람 수": [
        total_count,
        missing_count
    ],
    "뇌졸중 사람 수": [
        total_stroke_count,
        missing_stroke_count
    ],
    "뇌졸중 비율 (%)": [
        round(total_rate, 2),
        round(missing_rate, 2)
    ]
})

st.dataframe(
    bmi_comparison,
    use_container_width=True,
    hide_index=True
)

if missing_count == 0:
    st.info("BMI가 비어 있는 사람이 없습니다.")


# ==========================================
# 5. 흡연 상태별 사람 수
# ==========================================

st.header("5. 흡연 상태별 사람 수")

st.write(
    """
    흡연 상태에 해당하는 값마다 사람 수를 세어 봅니다.
    """
)

smoking_data = df.copy()

smoking_data["smoking_status"] = (
    smoking_data["smoking_status"]
    .fillna("결측")
)

smoking_count = (
    smoking_data["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

st.dataframe(
    smoking_count,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "결측은 흡연 상태 정보가 입력되지 않은 경우입니다."
)

# ==========================================
# 마무리
# ==========================================

st.divider()

st.caption(
    "※ 이 페이지는 데이터의 분포와 그룹별 차이를 탐색하기 위한 것입니다. "
    "그래프에서 관찰되는 차이만으로 특정 요인이 뇌졸중의 원인이라고 "
    "단정할 수는 없습니다."
)
