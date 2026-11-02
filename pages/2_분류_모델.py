
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="분류 모델 | 뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/Lin-1120/data/refs/heads/main/stroke.csv"

RANDOM_STATE = 42

# 열 이름과 우리말 이름 연결
FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

LABEL_TO_COLUMN = {
    value: key
    for key, value in FEATURE_NAMES.items()
}


# ==========================================
# 2. 화면 제목
# ==========================================

st.title("🧠 분류 모델")
st.caption(
    "건강 정보를 바탕으로 뇌졸중 여부를 예측하는 "
    "두 가지 분류 모델을 비교합니다."
)


# ==========================================
# 3. 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():

    return pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )


try:

    df = load_data()

except Exception as error:

    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(error))
    st.stop()


required_columns = [
    "id",
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
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


# 숫자형으로 변환
for col in required_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# 뇌졸중 여부가 확인되는 행만 사용
df = df.dropna(
    subset=["id", "stroke"]
).copy()

df = df[df["stroke"].isin([0, 1])].copy()

df["stroke"] = df["stroke"].astype(int)

# 번호 순으로 정렬
df = df.sort_values(
    "id",
    ascending=True
).reset_index(drop=True)


# ==========================================
# 4. 입력 속성 선택
# ==========================================

st.header("1. 예측에 사용할 속성 선택")

st.write(
    """
    뇌졸중 여부를 예측하는 데 사용할 건강 정보를 선택하세요.
    처음에는 체질량지수(BMI)를 제외한 네 가지 속성이 선택되어 있습니다.
    """
)

default_features = [
    "나이",
    "평균 혈당",
    "고혈압",
    "심장병"
]

selected_names = st.multiselect(
    "입력 속성",
    options=list(FEATURE_NAMES.values()),
    default=default_features
)

selected_features = [
    LABEL_TO_COLUMN[name]
    for name in selected_names
]

if len(selected_features) < 2:

    st.warning(
        "분류 모델을 만들려면 입력 속성을 두 개 이상 선택해야 합니다."
    )

    st.stop()


st.write(
    "**선택한 속성:** "
    + ", ".join(selected_names)
)


# ==========================================
# 5. 학습 데이터와 테스트 데이터 나누기
# ==========================================

st.header("2. 학습 데이터와 테스트 데이터")

st.write(
    """
    데이터를 번호 순으로 정렬한 뒤 열 명씩 묶습니다.

    각 묶음의 앞 세 명은 테스트 데이터로,
    나머지 일곱 명은 학습 데이터로 사용합니다.
    """
)

# 10명씩 묶었을 때 앞의 3명인지 확인
positions = pd.Series(
    range(len(df)),
    index=df.index
)

test_mask = (positions % 10) < 3

test_df = df.loc[test_mask].copy()
train_df = df.loc[~test_mask].copy()

# BMI가 선택된 경우에만 학습 데이터의 중앙값으로 채우기
bmi_median = None

if "bmi" in selected_features:

    bmi_median = train_df["bmi"].median()

    if pd.isna(bmi_median):

        st.error(
            "학습 데이터에서 체질량지수(BMI)의 중앙값을 구할 수 없습니다."
        )

        st.stop()

    train_df.loc[:, "bmi"] = (
        train_df["bmi"].fillna(bmi_median)
    )

    test_df.loc[:, "bmi"] = (
        test_df["bmi"].fillna(bmi_median)
    )

# BMI 이외 선택 속성에 결측치가 있는지 확인
other_features = [
    col for col in selected_features
    if col != "bmi"
]

if train_df[other_features].isna().any().any():

    st.error(
        "BMI 이외의 선택 속성에 빈 값이 있습니다. "
        "현재 실습에서는 BMI만 결측치 처리를 하도록 설정되어 있습니다."
    )

    st.stop()

if test_df[selected_features].isna().any().any():

    st.error(
        "테스트 데이터에 처리되지 않은 빈 값이 있습니다."
    )

    st.stop()


X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].copy()
y_test = test_df["stroke"].copy()


# 훈련용과 테스트용 데이터 수
train_count = len(train_df)
test_count = len(test_df)

col1, col2, col3 = st.columns(3)

col1.metric(
    "전체 데이터",
    f"{len(df):,}명"
)

col2.metric(
    "훈련 데이터",
    f"{train_count:,}명"
)

col3.metric(
    "테스트 데이터",
    f"{test_count:,}명"
)

if test_count != 1533:

    st.warning(
        f"현재 테스트 데이터는 {test_count:,}명입니다. "
        "요청한 1,533명과 다르므로 CSV의 전체 행 수를 확인해 주세요."
    )

else:

    st.success(
        "테스트 데이터가 요청한 1,533명으로 분리되었습니다."
    )


if train_count == 0 or test_count == 0:

    st.error("훈련 데이터 또는 테스트 데이터가 없습니다.")

    st.stop()


# ==========================================
# 6. BMI 결측치 처리 설명
# ==========================================

if "bmi" in selected_features:

    st.info(
        f"체질량지수(BMI)의 빈 값은 "
        f"훈련 데이터의 중앙값인 {bmi_median:.2f}로 채웠습니다. "
        "테스트 데이터에도 같은 값을 적용했습니다."
    )

else:

    st.info(
        "체질량지수(BMI)를 선택하지 않았으므로 "
        "BMI 결측치 처리는 하지 않았습니다."
    )


# ==========================================
# 7. 모델 만들기
# ==========================================

st.header("3. 두 가지 분류 모델 만들기")

st.write(
    """
    두 모델은 같은 훈련 데이터와 테스트 데이터를 사용합니다.

    뇌졸중(stroke)이 1이면 양성, 0이면 음성으로 분류합니다.
    """
)


# ------------------------------------------
# 모델 A: 로지스틱 회귀
# ------------------------------------------

# 크기 맞추기는 훈련 데이터만으로 진행
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=2000
)

logistic_model.fit(
    X_train_scaled,
    y_train
)


# ------------------------------------------
# 모델 B: 의사결정트리
# ------------------------------------------

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE
)

tree_model.fit(
    X_train,
    y_train
)


# ==========================================
# 8. 정확도 계산
# ==========================================

# 로지스틱 회귀
logistic_train_accuracy = logistic_model.score(
    X_train_scaled,
    y_train
)

logistic_test_accuracy = logistic_model.score(
    X_test_scaled,
    y_test
)


# 의사결정트리
tree_train_accuracy = tree_model.score(
    X_train,
    y_train
)

tree_test_accuracy = tree_model.score(
    X_test,
    y_test
)


# ------------------------------------------
# 아무 입력도 보지 않는 기준 모델
# ------------------------------------------

# 훈련 데이터에서 더 많은 쪽을 선택
majority_class = int(y_train.value_counts().idxmax())

baseline_train_accuracy = (
    y_train == majority_class
).mean()

baseline_test_accuracy = (
    y_test == majority_class
).mean()

majority_name = (
    "뇌졸중" if majority_class == 1
    else "뇌졸중 아님"
)


# ==========================================
# 9. 정확도 카드
# ==========================================

st.header("4. 모델의 정확도 비교")

st.write(
    """
    테스트 데이터의 정확도는 모델이 학습에 사용하지 않은 사람들에게
    얼마나 정확하게 답했는지를 나타냅니다.
    """
)


def show_accuracy_card(
    title,
    test_accuracy,
    train_accuracy,
    description
):

    with st.container(border=True):

        st.markdown(f"**{title}**")

        st.metric(
            "테스트 데이터 정확도",
            f"{test_accuracy * 100:.2f}%"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.caption("훈련 데이터")

            st.markdown(
                f"### {train_accuracy * 100:.2f}%"
            )

        with col2:

            st.caption("테스트 데이터")

            st.markdown(
                f"### {test_accuracy * 100:.2f}%"
            )

        st.caption(description)


col1, col2, col3 = st.columns(3)

with col1:

    show_accuracy_card(
        "로지스틱 회귀(확률로 답하는 모델)",
        logistic_test_accuracy,
        logistic_train_accuracy,
        "확률을 계산한 뒤 0.5를 기준으로 분류합니다."
    )

with col2:

    show_accuracy_card(
        "의사결정트리(질문으로 답하는 모델)",
        tree_test_accuracy,
        tree_train_accuracy,
        "최대 세 단계의 질문을 통해 분류합니다."
    )

with col3:

    show_accuracy_card(
        "기준 모델(입력을 보지 않는 모델)",
        baseline_test_accuracy,
        baseline_train_accuracy,
        f"훈련 데이터에서 더 많은 쪽인 '{majority_name}'만 답합니다."
    )


# ==========================================
# 10. 산점도에 사용할 두 속성 선택
# ==========================================

st.header("5. 두 속성으로 살펴보는 분류 결과")

st.write(
    """
    선택한 입력 속성 가운데 두 개를 골라 가로축과 세로축으로 사용합니다.

    점은 테스트 데이터의 각 사람을 나타내며,
    실제 뇌졸중 여부에 따라 색을 구분합니다.
    """
)

axis_col1, axis_col2 = st.columns(2)

with axis_col1:

    x_name = st.selectbox(
        "가로축 속성",
        options=selected_names,
        index=0
    )

with axis_col2:

    y_default_index = 1

    y_name = st.selectbox(
        "세로축 속성",
        options=selected_names,
        index=y_default_index
    )

x_feature = LABEL_TO_COLUMN[x_name]
y_feature = LABEL_TO_COLUMN[y_name]

if x_feature == y_feature:

    st.warning(
        "가로축과 세로축은 서로 다른 속성을 선택해야 합니다."
    )

    st.stop()


# ==========================================
# 11. 나머지 속성을 고정한 예측 평면
# ==========================================

# 두 축 이외의 속성은 테스트 데이터 중앙값으로 고정
fixed_values = {}

for feature in selected_features:

    if feature not in [x_feature, y_feature]:

        fixed_values[feature] = float(
            X_test[feature].median()
        )


if fixed_values:

    st.write("**그림에서 고정한 나머지 속성의 값**")

    fixed_text = []

    for feature, value in fixed_values.items():

        fixed_text.append(
            f"{FEATURE_NAMES[feature]}: {value:.2f}"
        )

    st.write(" · ".join(fixed_text))

else:

    st.write(
        "선택한 속성이 두 개뿐이므로 고정할 나머지 속성이 없습니다."
    )


# ==========================================
# 12. 그림을 그릴 범위 만들기
# ==========================================

x_min = float(X_test[x_feature].min())
x_max = float(X_test[x_feature].max())

y_min = float(X_test[y_feature].min())
y_max = float(X_test[y_feature].max())

x_padding = (x_max - x_min) * 0.08
y_padding = (y_max - y_min) * 0.08

if x_padding == 0:
    x_padding = 1

if y_padding == 0:
    y_padding = 1

plot_x_min = x_min - x_padding
plot_x_max = x_max + x_padding

plot_y_min = y_min - y_padding
plot_y_max = y_max + y_padding

grid_size = 81

x_values = [
    plot_x_min
    + i * (plot_x_max - plot_x_min) / (grid_size - 1)
    for i in range(grid_size)
]

y_values = [
    plot_y_min
    + i * (plot_y_max - plot_y_min) / (grid_size - 1)
    for i in range(grid_size)
]


# 각 좌표에서 나머지 속성은 고정값 사용
grid_rows = []

for y_value in y_values:

    for x_value in x_values:

        row = {}

        for feature in selected_features:

            if feature == x_feature:

                row[feature] = x_value

            elif feature == y_feature:

                row[feature] = y_value

            else:

                row[feature] = fixed_values[feature]

        grid_rows.append(row)


grid_df = pd.DataFrame(
    grid_rows,
    columns=selected_features
)


# ==========================================
# 13. 로지스틱 회귀의 0.5 경계선
# ==========================================

grid_scaled = scaler.transform(grid_df)

grid_probabilities = logistic_model.predict_proba(
    grid_scaled
)[:, 1]

probability_matrix = [
    grid_probabilities[
        i * grid_size:(i + 1) * grid_size
    ].tolist()
    for i in range(grid_size)
]


# ==========================================
# 14. 의사결정트리의 영역
# ==========================================

tree_grid_predictions = tree_model.predict(
    grid_df
)

tree_matrix = [
    tree_grid_predictions[
        i * grid_size:(i + 1) * grid_size
    ].tolist()
    for i in range(grid_size)
]


# ==========================================
# 15. Plotly 산점도
# ==========================================

plot_df = X_test.copy()

plot_df["실제 결과"] = y_test.map({
    0: "뇌졸중 아님",
    1: "뇌졸중"
})

plot_df["실제 결과"] = pd.Categorical(
    plot_df["실제 결과"],
    categories=["뇌졸중 아님", "뇌졸중"],
    ordered=True
)

fig = go.Figure()


# ------------------------------------------
# 의사결정트리의 예측 영역
# ------------------------------------------

fig.add_trace(
    go.Heatmap(
        x=x_values,
        y=y_values,
        z=tree_matrix,
        colorscale=[
            [0, "rgba(91,155,213,0.16)"],
            [1, "rgba(231,111,81,0.16)"]
        ],
        showscale=False,
        hoverinfo="skip",
        name="의사결정트리 영역"
    )
)


# ------------------------------------------
# 로지스틱 회귀의 0.5 경계선
# ------------------------------------------

fig.add_trace(
    go.Contour(
        x=x_values,
        y=y_values,
        z=probability_matrix,
        contours=dict(
            start=0.5,
            end=0.5,
            size=1,
            coloring="lines",
            showlabels=False
        ),
        line=dict(
            color="#222222",
            width=3
        ),
        showscale=False,
        hoverinfo="skip",
        name="로지스틱 회귀 0.5 경계선"
    )
)


# ------------------------------------------
# 실제 테스트 데이터 점
# ------------------------------------------

for label, color in [
    ("뇌졸중 아님", "#2878B5"),
    ("뇌졸중", "#E05252")
]:

    group = plot_df[
        plot_df["실제 결과"] == label
    ]

    fig.add_trace(
        go.Scatter(
            x=group[x_feature],
            y=group[y_feature],
            mode="markers",
            name=label,
            marker=dict(
                color=color,
                size=7,
                opacity=0.75,
                line=dict(
                    width=0.5,
                    color="white"
                )
            ),
            hovertemplate=(
                f"{x_name}: %{{x:.2f}}<br>"
                f"{y_name}: %{{y:.2f}}<br>"
                f"실제 결과: {label}"
                "<extra></extra>"
            )
        )
    )


fig.update_layout(
    title=(
        "테스트 데이터와 두 분류 모델의 예측 영역"
    ),
    xaxis_title=x_name,
    yaxis_title=y_name,
    template="plotly_white",
    height=650,
    legend_title="실제 뇌졸중 여부"
)

fig.update_xaxes(
    range=[plot_x_min, plot_x_max]
)

fig.update_yaxes(
    range=[plot_y_min, plot_y_max]
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==========================================
# 16. 경계선 표시 여부 확인
# ==========================================

min_probability = min(grid_probabilities)
max_probability = max(grid_probabilities)

if (
    min_probability > 0.5
    or max_probability < 0.5
):

    st.warning(
        "로지스틱 회귀의 0.5 경계선이 "
        "현재 그림의 범위 안에 나타나지 않습니다."
    )

else:

    st.success(
        "로지스틱 회귀의 0.5 경계선이 "
        "그림의 범위 안에 있습니다."
    )


st.caption(
    "옅은 파란색과 붉은색 영역은 의사결정트리의 예측 영역입니다. "
    "검은 선은 로지스틱 회귀가 뇌졸중 확률 0.5를 기준으로 나누는 경계선입니다. "
    "점의 색은 모델의 예측이 아닌 실제 뇌졸중 여부를 나타냅니다."
)


# ==========================================
# 17. 의사결정트리의 질문 구조
# ==========================================

st.header("6. 의사결정트리는 어떤 질문을 했을까?")

st.write(
    """
    의사결정트리는 질문에 대한 답에 따라 가지를 나누는 모델입니다.

    각 질문은 한 번에 한 가지 속성을 기준으로 합니다.
    최대 세 번까지만 질문하도록 설정했습니다.
    """
)


# ==========================================
# 18. 각 마디에 도달한 실제 훈련 인원 계산
# ==========================================

decision_path = tree_model.decision_path(
    X_train
)

tree_structure = tree_model.tree_

node_count = tree_structure.node_count

node_total = [0] * node_count
node_positive = [0] * node_count

# 각 훈련 데이터가 지나간 마디에 인원수 누적
for row_index in range(decision_path.shape[0]):

    start = decision_path.indptr[row_index]
    end = decision_path.indptr[row_index + 1]

    visited_nodes = decision_path.indices[start:end]

    actual_label = int(y_train.iloc[row_index])

    for node in visited_nodes:

        node_total[node] += 1

        if actual_label == 1:

            node_positive[node] += 1


# ==========================================
# 19. DOT 문자열 만들기
# ==========================================

dot_lines = [
    "digraph DecisionTree {",
    'graph [rankdir=TB, bgcolor="white", nodesep=0.35, ranksep=0.55];',
    'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=11, margin="0.15,0.10"];',
    'edge [fontname="Arial", fontsize=10, color="#666666"];'
]

leaf_count = 0
negative_leaf_count = 0
used_features = set()


def make_node_label(node):

    total = node_total[node]
    positive = node_positive[node]

    negative = total - positive

    rate = (
        positive / total * 100
        if total > 0 else 0
    )

    return (
        f"도착 인원: {total}명\\n"
        f"뇌졸중: {positive}명\\n"
        f"뇌졸중 비율: {rate:.1f}%"
    )


def add_tree_node(node):

    global leaf_count
    global negative_leaf_count

    left = tree_structure.children_left[node]
    right = tree_structure.children_right[node]

    positive = node_positive[node]
    negative = node_total[node] - positive

    # 리프 노드인지 확인
    is_leaf = left == right

    if is_leaf:

        leaf_count += 1

        predicted_class = (
            1 if positive >= negative else 0
        )

        if predicted_class == 0:

            negative_leaf_count += 1

            answer = "답: 뇌졸중 아님"
            fill_color = "#DCEBFA"

        else:

            answer = "답: 뇌졸중"
            fill_color = "#F9DADA"

        label = (
            answer
            + "\\n"
            + make_node_label(node)
        )

    else:

        feature_index = tree_structure.feature[node]

        feature = selected_features[feature_index]

        used_features.add(feature)

        threshold = tree_structure.threshold[node]

        question = (
            f"{FEATURE_NAMES[feature]} ≤ {threshold:.2f}?"
        )

        label = (
            question
            + "\\n"
            + make_node_label(node)
        )

        fill_color = "#F1F3F5"

    dot_lines.append(
        f'n{node} [label="{label}", fillcolor="{fill_color}"];'
    )

    if not is_leaf:

        feature_index = tree_structure.feature[node]

        threshold = tree_structure.threshold[node]

        # 왼쪽 가지: 조건이 참
        add_tree_node(left)

        dot_lines.append(
            f'n{node} -> n{left} [label="예"];'
        )

        # 오른쪽 가지: 조건이 거짓
        add_tree_node(right)

        dot_lines.append(
            f'n{node} -> n{right} [label="아니요"];'
        )


add_tree_node(0)

dot_lines.append("}")

dot_string = "\n".join(dot_lines)


# ==========================================
# 20. 의사결정트리 그림 표시
# ==========================================

st.graphviz_chart(
    dot_string,
    use_container_width=True
)


# ==========================================
# 21. 의사결정트리 요약
# ==========================================

st.subheader("의사결정트리 요약")

st.write(
    f"**답을 내는 마디(리프 노드)는 모두 {leaf_count}칸이며, "
    f"그중 {negative_leaf_count}칸이 '뇌졸중 아님'이라고 답합니다.**"
)

if used_features:

    used_names = [
        FEATURE_NAMES[feature]
        for feature in selected_features
        if feature in used_features
    ]

    st.write(
        "**이 나무가 실제로 질문한 속성:** "
        + ", ".join(used_names)
    )

else:

    st.write(
        "**이 나무가 실제로 질문한 속성:** "
        "질문 없이 하나의 답만 내렸습니다."
    )


# ==========================================
# 22. 마무리 설명
# ==========================================

st.divider()

st.subheader("실습에서 생각해 볼 점")

st.markdown(
    """
    1. 두 모델 중 테스트 데이터의 정확도가 더 높은 모델은 무엇인가?
    2. 훈련 데이터와 테스트 데이터의 정확도 차이는 어떤 의미일까?
    3. 입력 속성을 바꾸면 정확도와 분류 경계가 어떻게 달라질까?
    4. 의사결정트리가 실제로 사용한 질문은 무엇인가?
    5. 아무 입력도 보지 않는 기준 모델과 비교하면 어떤 점을 알 수 있을까?
    """
)

st.caption(
    "※ 이 실습은 교육용 데이터 분석입니다. "
    "모델의 예측 결과는 실제 의료 진단이나 개인의 뇌졸중 발생 가능성을 "
    "판정하는 데 사용할 수 없습니다."
)
