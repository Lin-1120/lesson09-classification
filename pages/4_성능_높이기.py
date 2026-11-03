
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="성능 높이기",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ 성능 높이기")
st.write(
    "뇌졸중을 겪은 사람이 드문 데이터에서 "
    "두 분류 모델의 가중치 적용 전과 후를 비교합니다."
)


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/Lin-1120/data/refs/heads/main/stroke.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 사용할 속성 선택
# --------------------------------------------------
FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

KOREAN_TO_ENGLISH = {
    value: key for key, value in FEATURE_NAMES.items()
}

st.subheader("1. 입력으로 사용할 속성 선택")

selected_names = st.multiselect(
    "모델이 뇌졸중 여부를 예측할 때 사용할 속성을 골라 주세요.",
    options=list(FEATURE_NAMES.values()),
    default=[
        "나이",
        "평균 혈당",
        "고혈압",
        "심장병"
    ]
)

if len(selected_names) < 2:
    st.warning("속성을 2개 이상 선택해 주세요.")
    st.stop()

selected_features = [
    KOREAN_TO_ENGLISH[name] for name in selected_names
]

st.caption(
    "속성을 3개 이상 선택하면 3차원 산점도를 표시합니다. "
    "3차원 그래프에서는 선택한 속성 중 3개를 축으로 사용합니다."
)


# --------------------------------------------------
# 데이터 전처리 및 분할
# 채점표 페이지와 같은 방식
# --------------------------------------------------
required_columns = [
    "id",
    "stroke",
    *selected_features
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        "데이터에 필요한 열이 없습니다: "
        + ", ".join(missing_columns)
    )
    st.stop()


# 숫자형 변환
for col in ["id", "stroke", *selected_features]:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# id와 stroke가 없는 행 제외
df = df.dropna(subset=["id", "stroke"])

# 뇌졸중 여부가 0 또는 1인 자료만 사용
df = df[df["stroke"].isin([0, 1])].copy()

# id 순서로 정렬
df = df.sort_values("id").reset_index(drop=True)

if len(df) < 10:
    st.error("데이터가 너무 적어 훈련용과 테스트용으로 나눌 수 없습니다.")
    st.stop()


# 10명씩 묶어 각 묶음의 앞 3명을 테스트용으로 사용
positions = np.arange(len(df))
test_mask = positions % 10 < 3

train_df = df.loc[~test_mask].copy()
test_df = df.loc[test_mask].copy()

if len(train_df) == 0 or len(test_df) == 0:
    st.error("훈련용 또는 테스트용 데이터가 없습니다.")
    st.stop()


# BMI를 선택했을 때만 훈련용 중앙값으로 결측값 채우기
if "bmi" in selected_features:
    bmi_median = train_df["bmi"].median()

    if pd.isna(bmi_median):
        st.error("훈련용 데이터의 체질량지수 중앙값을 구할 수 없습니다.")
        st.stop()

    train_df["bmi"] = train_df["bmi"].fillna(bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median)


# BMI 이외 선택 속성의 결측값 확인
other_features = [
    col for col in selected_features
    if col != "bmi"
]

if train_df[other_features].isna().any().any():
    st.error("BMI 이외의 훈련용 속성에 결측값이 있습니다.")
    st.stop()

if test_df[selected_features].isna().any().any():
    st.error("테스트용 속성에 결측값이 있습니다.")
    st.stop()


X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

if y_train.nunique() < 2:
    st.error("훈련용 데이터에 뇌졸중 여부가 한 종류만 있어 모델을 학습할 수 없습니다.")
    st.stop()


# --------------------------------------------------
# 모델 학습
# --------------------------------------------------
# 로지스틱 회귀: 표준화
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 가중치 적용 전
logistic_before = LogisticRegression(
    random_state=42,
    max_iter=2000
)

tree_before = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)


# 가중치 적용 후
logistic_after = LogisticRegression(
    random_state=42,
    max_iter=2000,
    class_weight="balanced"
)

tree_after = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced"
)


models = {
    "로지스틱 회귀(확률로 답하는 모델)": {
        "before": logistic_before,
        "after": logistic_after,
        "scale": True
    },
    "의사결정트리(질문으로 답하는 모델)": {
        "before": tree_before,
        "after": tree_after,
        "scale": False
    }
}


for model_info in models.values():
    if model_info["scale"]:
        model_info["before"].fit(X_train_scaled, y_train)
        model_info["after"].fit(X_train_scaled, y_train)
    else:
        model_info["before"].fit(X_train, y_train)
        model_info["after"].fit(X_train, y_train)


# --------------------------------------------------
# 예측 및 평가 함수
# --------------------------------------------------
def predict_probability(model, X):
    if isinstance(model, LogisticRegression):
        X_input = scaler.transform(X)
    else:
        X_input = X

    return model.predict_proba(X_input)[:, 1]


def get_confusion_values(y_true, y_pred):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


def calculate_metrics(values):
    tn = values["TN"]
    fp = values["FP"]
    fn = values["FN"]
    tp = values["TP"]

    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total else None
    recall = tp / (tp + fn) if (tp + fn) else None
    precision = tp / (tp + fp) if (tp + fp) else None

    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = None
    else:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    return {
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1 점수": f1
    }


def format_metric(value):
    if value is None:
        return "계산할 수 없음"
    return f"{value:.3f}"


# 모델별 예측 결과 저장
results = {}

for model_name, info in models.items():
    before_model = info["before"]
    after_model = info["after"]

    before_probability = predict_probability(
        before_model, X_test
    )

    after_probability = predict_probability(
        after_model, X_test
    )

    before_prediction = (
        before_probability >= 0.5
    ).astype(int)

    after_prediction = (
        after_probability >= 0.5
    ).astype(int)

    before_values = get_confusion_values(
        y_test, before_prediction
    )

    after_values = get_confusion_values(
        y_test, after_prediction
    )

    results[model_name] = {
        "before_model": before_model,
        "after_model": after_model,
        "before_probability": before_probability,
        "after_probability": after_probability,
        "before_prediction": before_prediction,
        "after_prediction": after_prediction,
        "before_values": before_values,
        "after_values": after_values,
        "before_metrics": calculate_metrics(before_values),
        "after_metrics": calculate_metrics(after_values)
    }


# --------------------------------------------------
# 평가 결과
# --------------------------------------------------
st.divider()
st.subheader("2. 가중치 적용 전과 후의 성능 비교")

test_count = len(test_df)
test_positive_count = int(y_test.sum())

st.markdown(
    f"**테스트 데이터: {test_count:,}명 / "
    f"그중 실제 뇌졸중인 사람: {test_positive_count:,}명**"
)

st.caption(
    "가중치 적용 후에는 뇌졸중 사례를 더 중요하게 학습하도록 설정했습니다. "
    "테스트 데이터는 두 모델 모두 동일하게 사용합니다."
)


def make_comparison_table(result):
    before = result["before_values"]
    after = result["after_values"]

    before_metrics = result["before_metrics"]
    after_metrics = result["after_metrics"]

    rows = [
        ("참음성(TN)", before["TN"], after["TN"]),
        ("거짓양성(FP)", before["FP"], after["FP"]),
        ("거짓음성(FN)", before["FN"], after["FN"]),
        ("참양성(TP)", before["TP"], after["TP"]),
        (
            "정확도",
            format_metric(before_metrics["정확도"]),
            format_metric(after_metrics["정확도"])
        ),
        (
            "재현율",
            format_metric(before_metrics["재현율"]),
            format_metric(after_metrics["재현율"])
        ),
        (
            "정밀도",
            format_metric(before_metrics["정밀도"]),
            format_metric(after_metrics["정밀도"])
        ),
        (
            "F1 점수",
            format_metric(before_metrics["F1 점수"]),
            format_metric(after_metrics["F1 점수"])
        )
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "평가 항목",
            "가중치 적용 전",
            "가중치 적용 후"
        ]
    )


for model_name, result in results.items():
    st.markdown(f"### {model_name}")
    st.dataframe(
        make_comparison_table(result),
        hide_index=True,
        use_container_width=True
    )


# --------------------------------------------------
# 환자 수 변화 요약
# --------------------------------------------------
st.subheader("3. 뇌졸중 예측 결과의 변화")

summary_lines = []

for model_name, result in results.items():
    before = result["before_values"]
    after = result["after_values"]

    summary_lines.append(
        f"**{model_name}** — "
        f"찾아낸 환자 수(TP): {before['TP']}명 → {after['TP']}명, "
        f"놓친 환자 수(FN): {before['FN']}명 → {after['FN']}명, "
        f"헛짚은 사람 수(FP): {before['FP']}명 → {after['FP']}명"
    )

for line in summary_lines:
    st.markdown(line)


# --------------------------------------------------
# 그래프에 사용할 축 선택
# --------------------------------------------------
st.divider()
st.subheader("4. 실제 데이터와 0.5 경계선 비교")

st.write(
    "점은 테스트 데이터의 실제 뇌졸중 여부를 나타냅니다. "
    "경계선 또는 평면은 모델이 뇌졸중 확률을 0.5로 판단하는 위치입니다."
)

if len(selected_features) == 2:
    axis_features = selected_features

else:
    st.write("3차원 그래프의 축으로 사용할 속성을 선택하세요.")

    axis_col1, axis_col2, axis_col3 = st.columns(3)

    with axis_col1:
        x_feature = st.selectbox(
            "가로축(X)",
            selected_features,
            index=0,
            format_func=lambda x: FEATURE_NAMES[x]
        )

    remaining_y = [
        feature for feature in selected_features
        if feature != x_feature
    ]

    with axis_col2:
        y_feature = st.selectbox(
            "세로축(Y)",
            remaining_y,
            index=0,
            format_func=lambda x: FEATURE_NAMES[x]
        )

    remaining_z = [
        feature for feature in selected_features
        if feature not in [x_feature, y_feature]
    ]

    with axis_col3:
        z_feature = st.selectbox(
            "깊이축(Z)",
            remaining_z,
            index=0,
            format_func=lambda x: FEATURE_NAMES[x]
        )

    axis_features = [
        x_feature,
        y_feature,
        z_feature
    ]


# --------------------------------------------------
# 그래프용 예측 함수
# 선택하지 않은 속성은 훈련용 중앙값으로 고정
# --------------------------------------------------
fixed_values = X_train.median()


def make_prediction_grid(grid_df, model):
    full_grid = pd.DataFrame(
        np.tile(
            fixed_values.values,
            (len(grid_df), 1)
        ),
        columns=selected_features
    )

    for col in grid_df.columns:
        full_grid[col] = grid_df[col].to_numpy()

    return predict_probability(model, full_grid)


# --------------------------------------------------
# 2차원 산점도 및 경계선
# --------------------------------------------------
if len(selected_features) == 2:

    x_feature, y_feature = axis_features

    x_min = float(X_test[x_feature].min())
    x_max = float(X_test[x_feature].max())
    y_min = float(X_test[y_feature].min())
    y_max = float(X_test[y_feature].max())

    if x_min == x_max or y_min == y_max:
        st.warning("선택한 속성의 값이 모두 같아 산점도를 그릴 수 없습니다.")
        st.stop()

    x_range = np.linspace(x_min, x_max, 90)
    y_range = np.linspace(y_min, y_max, 90)

    xx, yy = np.meshgrid(x_range, y_range)

    grid_df = pd.DataFrame({
        x_feature: xx.ravel(),
        y_feature: yy.ravel()
    })

    fig = go.Figure()

    # 실제 뇌졸중 여부에 따른 테스트 데이터 점
    for actual_value, label, color in [
        (0, "실제 뇌졸중 아님", "#3977C3"),
        (1, "실제 뇌졸중", "#D94B4B")
    ]:
        subset = test_df[
            test_df["stroke"] == actual_value
        ]

        fig.add_trace(
            go.Scatter(
                x=subset[x_feature],
                y=subset[y_feature],
                mode="markers",
                name=label,
                marker=dict(
                    color=color,
                    size=7,
                    opacity=0.65
                )
            )
        )

    boundary_messages = []

    boundary_styles = [
        (
            "로지스틱 회귀(확률로 답하는 모델)",
            "before_model",
            "가중치 적용 전",
            "#E69F00"
        ),
        (
            "로지스틱 회귀(확률로 답하는 모델)",
            "after_model",
            "가중치 적용 후",
            "#CC79A7"
        ),
        (
            "의사결정트리(질문으로 답하는 모델)",
            "before_model",
            "가중치 적용 전",
            "#009E73"
        ),
        (
            "의사결정트리(질문으로 답하는 모델)",
            "after_model",
            "가중치 적용 후",
            "#56B4E9"
        )
    ]

    for model_name, model_key, stage, color in boundary_styles:
        model = results[model_name][model_key]

        probabilities = make_prediction_grid(
            grid_df,
            model
        ).reshape(xx.shape)

        if probabilities.min() <= 0.5 <= probabilities.max():
            fig.add_trace(
                go.Contour(
                    x=x_range,
                    y=y_range,
                    z=probabilities,
                    contours=dict(
                        start=0.5,
                        end=0.5,
                        size=0.01,
                        coloring="lines"
                    ),
                    line=dict(
                        color=color,
                        width=3
                    ),
                    showscale=False,
                    hoverinfo="skip",
                    name=f"{model_name} - {stage}"
                )
            )
        else:
            boundary_messages.append(
                f"{model_name}의 {stage} 0.5 경계선은 "
                "현재 그래프 범위 안에 없습니다."
            )

    fig.update_layout(
        title="실제 뇌졸중 여부와 모델별 0.5 경계선",
        xaxis_title=FEATURE_NAMES[x_feature],
        yaxis_title=FEATURE_NAMES[y_feature],
        height=700,
        legend_title="점과 경계선",
        hovermode="closest"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    for message in boundary_messages:
        st.info(message)


# --------------------------------------------------
# 3차원 산점도 및 경계 평면
# --------------------------------------------------
else:

    x_feature, y_feature, z_feature = axis_features

    x_min = float(X_test[x_feature].min())
    x_max = float(X_test[x_feature].max())
    y_min = float(X_test[y_feature].min())
    y_max = float(X_test[y_feature].max())
    z_min = float(X_test[z_feature].min())
    z_max = float(X_test[z_feature].max())

    if (
        x_min == x_max
        or y_min == y_max
        or z_min == z_max
    ):
        st.warning("선택한 축 중 값이 모두 같은 속성이 있어 3차원 그래프를 그릴 수 없습니다.")
        st.stop()

    x_range = np.linspace(x_min, x_max, 22)
    y_range = np.linspace(y_min, y_max, 22)
    z_range = np.linspace(z_min, z_max, 22)

    xx, yy, zz = np.meshgrid(
        x_range,
        y_range,
        z_range,
        indexing="ij"
    )

    grid_df = pd.DataFrame({
        x_feature: xx.ravel(),
        y_feature: yy.ravel(),
        z_feature: zz.ravel()
    })

    fig = go.Figure()

    # 실제 뇌졸중 여부에 따른 테스트 데이터 점
    for actual_value, label, color in [
        (0, "실제 뇌졸중 아님", "#3977C3"),
        (1, "실제 뇌졸중", "#D94B4B")
    ]:
        subset = test_df[
            test_df["stroke"] == actual_value
        ]

        fig.add_trace(
            go.Scatter3d(
                x=subset[x_feature],
                y=subset[y_feature],
                z=subset[z_feature],
                mode="markers",
                name=label,
                marker=dict(
                    color=color,
                    size=4,
                    opacity=0.7
                )
            )
        )

    boundary_messages = []

    boundary_styles = [
        (
            "로지스틱 회귀(확률로 답하는 모델)",
            "before_model",
            "가중치 적용 전",
            "Reds"
        ),
        (
            "로지스틱 회귀(확률로 답하는 모델)",
            "after_model",
            "가중치 적용 후",
            "Purples"
        ),
        (
            "의사결정트리(질문으로 답하는 모델)",
            "before_model",
            "가중치 적용 전",
            "Greens"
        ),
        (
            "의사결정트리(질문으로 답하는 모델)",
            "after_model",
            "가중치 적용 후",
            "Oranges"
        )
    ]

    for model_name, model_key, stage, colorscale in boundary_styles:
        model = results[model_name][model_key]

        probabilities = make_prediction_grid(
            grid_df,
            model
        )

        if probabilities.min() <= 0.5 <= probabilities.max():
            fig.add_trace(
                go.Isosurface(
                    x=xx.ravel(),
                    y=yy.ravel(),
                    z=zz.ravel(),
                    value=probabilities,
                    isomin=0.5,
                    isomax=0.5,
                    surface_count=1,
                    colorscale=colorscale,
                    opacity=0.28,
                    showscale=False,
                    caps=dict(
                        x_show=False,
                        y_show=False,
                        z_show=False
                    ),
                    name=f"{model_name} - {stage}",
                    hoverinfo="skip"
                )
            )
        else:
            boundary_messages.append(
                f"{model_name}의 {stage} 0.5 경계 평면은 "
                "현재 그래프 범위 안에 없습니다."
            )

    fig.update_layout(
        title="실제 뇌졸중 여부와 모델별 0.5 경계 평면",
        scene=dict(
            xaxis_title=FEATURE_NAMES[x_feature],
            yaxis_title=FEATURE_NAMES[y_feature],
            zaxis_title=FEATURE_NAMES[z_feature]
        ),
        height=800,
        legend_title="점과 경계 평면"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    for message in boundary_messages:
        st.info(message)


# --------------------------------------------------
# 해석 안내
# --------------------------------------------------
st.divider()
st.subheader("5. 결과 해석하기")

st.markdown(
    """
    - **참양성(TP)**: 실제 뇌졸중 사례를 모델이 뇌졸중이라고 맞힌 사람
    - **거짓음성(FN)**: 실제 뇌졸중 사례를 모델이 놓친 사람
    - **거짓양성(FP)**: 실제 뇌졸중이 아닌 사람을 뇌졸중이라고 예측한 사람
    - **재현율**: 실제 뇌졸중 사례 중 모델이 찾아낸 비율
    - **정밀도**: 모델이 뇌졸중이라고 예측한 사람 중 실제 뇌졸중인 비율
    - **F1 점수**: 정밀도와 재현율을 함께 고려한 지표
    """
)

st.info(
    "가중치를 적용하면 뇌졸중 사례를 더 잘 찾아낼 수 있지만, "
    "뇌졸중이 아닌 사람을 뇌졸중이라고 예측하는 경우도 늘어날 수 있습니다. "
    "따라서 정확도 하나만 보지 말고 재현율, 정밀도, F1 점수와 "
    "거짓양성·거짓음성의 변화를 함께 비교해야 합니다."
)

st.caption(
    "이 앱은 분류 알고리즘의 학습과 평가를 위한 교육용 실습입니다. "
    "실제 환자의 뇌졸중 진단이나 의료적 판단에 사용할 수 없습니다."
)
