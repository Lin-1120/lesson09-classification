
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="채점표 | 뇌졸중 예측 실습실",
    page_icon="📝",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/Lin-1120/data/refs/heads/main/stroke.csv"

RANDOM_STATE = 42

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

st.title("📝 채점표")
st.caption(
    "두 분류 모델의 예측 결과를 혼동행렬과 평가 지표로 비교합니다."
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


# ==========================================
# 4. 데이터 전처리
# ==========================================

for col in required_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# 번호와 뇌졸중 여부가 확인되는 데이터만 사용
df = df.dropna(
    subset=["id", "stroke"]
).copy()

df = df[
    df["stroke"].isin([0, 1])
].copy()

df["stroke"] = df["stroke"].astype(int)


# 번호 순으로 정렬
df = df.sort_values(
    "id",
    ascending=True
).reset_index(drop=True)


# ==========================================
# 5. 입력 속성 선택
# ==========================================

st.header("1. 예측에 사용할 속성")

st.write(
    """
    분류 모델 페이지와 동일하게 예측에 사용할 속성을 선택합니다.
    처음에는 BMI를 제외한 네 가지 속성이 선택되어 있습니다.
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
# 6. 훈련 데이터와 테스트 데이터 분리
# ==========================================

st.header("2. 훈련 데이터와 테스트 데이터")

st.write(
    """
    번호 순으로 정렬한 뒤 열 명씩 묶습니다.

    각 묶음의 앞 세 명은 테스트 데이터,
    나머지 일곱 명은 훈련 데이터로 사용합니다.
    """
)

positions = pd.Series(
    range(len(df)),
    index=df.index
)

test_mask = (positions % 10) < 3

test_df = df.loc[test_mask].copy()
train_df = df.loc[~test_mask].copy()


# ==========================================
# 7. BMI 결측치 처리
# ==========================================

bmi_median = None

if "bmi" in selected_features:

    bmi_median = train_df["bmi"].median()

    if pd.isna(bmi_median):

        st.error(
            "훈련 데이터에서 BMI 중앙값을 구할 수 없습니다."
        )

        st.stop()

    train_df.loc[:, "bmi"] = (
        train_df["bmi"].fillna(bmi_median)
    )

    test_df.loc[:, "bmi"] = (
        test_df["bmi"].fillna(bmi_median)
    )


# BMI 이외의 속성에 결측치가 있는지 확인
other_features = [
    feature
    for feature in selected_features
    if feature != "bmi"
]

if train_df[other_features].isna().any().any():

    st.error(
        "훈련 데이터의 선택 속성에 빈 값이 있습니다."
    )

    st.stop()


if test_df[selected_features].isna().any().any():

    st.error(
        "테스트 데이터의 선택 속성에 빈 값이 있습니다."
    )

    st.stop()


# ==========================================
# 8. X와 y 만들기
# ==========================================

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].copy()
y_test = test_df["stroke"].copy()


if len(X_train) == 0 or len(X_test) == 0:

    st.error("훈련 데이터 또는 테스트 데이터가 없습니다.")

    st.stop()


if y_train.nunique() < 2:

    st.error(
        "훈련 데이터에 뇌졸중인 사람과 아닌 사람이 모두 있어야 "
        "분류 모델을 학습할 수 있습니다."
    )

    st.stop()


# ==========================================
# 9. 테스트 데이터 정보
# ==========================================

test_count = len(test_df)

test_positive_count = int(
    (y_test == 1).sum()
)

train_count = len(train_df)

st.markdown(
    f"""
    **테스트 데이터: {test_count:,}명 / "
    f"그중 실제 뇌졸중인 사람: {test_positive_count:,}명**
    """
)

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

if test_count == 1533:

    st.success(
        "테스트 데이터가 요청한 1,533명으로 분리되었습니다."
    )

else:

    st.warning(
        f"현재 테스트 데이터는 {test_count:,}명입니다. "
        "원하는 1,533명과 다르면 원본 데이터의 행 수를 확인하세요."
    )


if "bmi" in selected_features:

    st.info(
        f"BMI의 빈 값은 훈련 데이터의 중앙값 "
        f"{bmi_median:.2f}으로 채웠습니다."
    )

else:

    st.info(
        "BMI를 선택하지 않았으므로 BMI 결측치 처리를 하지 않았습니다."
    )


# ==========================================
# 10. 로지스틱 회귀 모델 학습
# ==========================================

st.header("3. 두 모델 학습")

# 훈련 데이터만으로 크기 맞추기
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


logistic_model = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=2000
)

logistic_model.fit(
    X_train_scaled,
    y_train
)


# ==========================================
# 11. 의사결정트리 모델 학습
# ==========================================

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
# 12. 테스트 데이터 예측
# ==========================================

# 로지스틱 회귀
logistic_train_pred = logistic_model.predict(
    X_train_scaled
)

logistic_test_pred = logistic_model.predict(
    X_test_scaled
)


# 의사결정트리
tree_train_pred = tree_model.predict(
    X_train
)

tree_test_pred = tree_model.predict(
    X_test
)


# ==========================================
# 13. 기준 모델
# ==========================================

# 훈련 데이터에서 사람이 더 많은 쪽을 선택
majority_class = int(
    y_train.value_counts().idxmax()
)

baseline_train_pred = [
    majority_class
] * len(y_train)

baseline_test_pred = [
    majority_class
] * len(y_test)


# ==========================================
# 14. 혼동행렬 계산 함수
# ==========================================

def make_confusion_values(actual, predicted):

    actual = list(actual)
    predicted = list(predicted)

    # 실제 뇌졸중 아님, 예측 뇌졸중 아님
    tn = sum(
        a == 0 and p == 0
        for a, p in zip(actual, predicted)
    )

    # 실제 뇌졸중 아님, 예측 뇌졸중
    fp = sum(
        a == 0 and p == 1
        for a, p in zip(actual, predicted)
    )

    # 실제 뇌졸중, 예측 뇌졸중 아님
    fn = sum(
        a == 1 and p == 0
        for a, p in zip(actual, predicted)
    )

    # 실제 뇌졸중, 예측 뇌졸중
    tp = sum(
        a == 1 and p == 1
        for a, p in zip(actual, predicted)
    )

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


# ==========================================
# 15. 혼동행렬 표시 함수
# ==========================================

def show_confusion_matrix(values):

    tn = values["TN"]
    fp = values["FP"]
    fn = values["FN"]
    tp = values["TP"]

    # 행 합계
    actual_negative = tn + fp
    actual_positive = fn + tp

    # 열 합계
    predicted_negative = tn + fn
    predicted_positive = fp + tp

    total = tn + fp + fn + tp

    matrix = pd.DataFrame(
        [
            [
                f"{tn:,}명 (TN)",
                f"{fp:,}명 (FP)",
                f"{actual_negative:,}명"
            ],
            [
                f"{fn:,}명 (FN)",
                f"{tp:,}명 (TP)",
                f"{actual_positive:,}명"
            ],
            [
                f"{predicted_negative:,}명",
                f"{predicted_positive:,}명",
                f"{total:,}명"
            ]
        ],
        index=[
            "실제: 뇌졸중 아님",
            "실제: 뇌졸중",
            "열 합계"
        ],
        columns=[
            "예측: 뇌졸중 아님",
            "예측: 뇌졸중",
            "행 합계"
        ]
    )

    st.dataframe(
        matrix,
        use_container_width=True
    )

    st.caption(
        "TN: 실제 아님 / 예측 아님 · "
        "FP: 실제 아님 / 예측 뇌졸중 · "
        "FN: 실제 뇌졸중 / 예측 아님 · "
        "TP: 실제 뇌졸중 / 예측 뇌졸중"
    )


# ==========================================
# 16. 평가 지표 계산 함수
# ==========================================

def calculate_metrics(values):

    tn = values["TN"]
    fp = values["FP"]
    fn = values["FN"]
    tp = values["TP"]

    total = tn + fp + fn + tp

    # 정확도
    accuracy = (
        (tp + tn) / total
        if total > 0
        else None
    )

    # 재현율: 실제 뇌졸중인 사람 중 맞힌 비율
    actual_positive = tp + fn

    recall = (
        tp / actual_positive
        if actual_positive > 0
        else None
    )

    # 정밀도: 뇌졸중이라고 예측한 사람 중 맞힌 비율
    predicted_positive = tp + fp

    if predicted_positive == 0:

        precision = None
        f1 = None

    else:

        precision = tp / predicted_positive

        if recall is None or (precision + recall) == 0:

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
        "F1": f1
    }


# ==========================================
# 17. 지표를 보기 좋은 형태로 변환
# ==========================================

def format_metric(value):

    if value is None:

        return "계산할 수 없음"

    return f"{value * 100:.2f}%"


# ==========================================
# 18. 혼동행렬 만들기
# ==========================================

logistic_values = make_confusion_values(
    y_test,
    logistic_test_pred
)

tree_values = make_confusion_values(
    y_test,
    tree_test_pred
)

baseline_values = make_confusion_values(
    y_test,
    baseline_test_pred
)


# ==========================================
# 19. 혼동행렬 표시
# ==========================================

st.header("4. 테스트 데이터 혼동행렬")

st.write(
    """
    혼동행렬은 실제 뇌졸중 여부와 모델의 예측 결과를
    교차해서 나타낸 표입니다.

    행은 실제 결과, 열은 예측 결과입니다.
    """
)

st.subheader(
    "로지스틱 회귀(확률로 답하는 모델)"
)

show_confusion_matrix(
    logistic_values
)


st.subheader(
    "의사결정트리(질문으로 답하는 모델)"
)

show_confusion_matrix(
    tree_values
)


st.subheader(
    "기준 모델(훈련용에서 많은 쪽으로만 답하는 모델)"
)

show_confusion_matrix(
    baseline_values
)


# ==========================================
# 20. 세 모델의 평가 지표 비교
# ==========================================

st.header("5. 평가 지표 비교")

st.write(
    """
    혼동행렬의 네 칸을 이용하여 정확도, 재현율,
    정밀도, F1을 계산합니다.
    """
)

logistic_metrics = calculate_metrics(
    logistic_values
)

tree_metrics = calculate_metrics(
    tree_values
)

baseline_metrics = calculate_metrics(
    baseline_values
)


metrics_table = pd.DataFrame(
    {
        "로지스틱 회귀(확률로 답하는 모델)": [
            format_metric(logistic_metrics["정확도"]),
            format_metric(logistic_metrics["재현율"]),
            format_metric(logistic_metrics["정밀도"]),
            format_metric(logistic_metrics["F1"])
        ],

        "의사결정트리(질문으로 답하는 모델)": [
            format_metric(tree_metrics["정확도"]),
            format_metric(tree_metrics["재현율"]),
            format_metric(tree_metrics["정밀도"]),
            format_metric(tree_metrics["F1"])
        ],

        "기준 모델(훈련용에서 많은 쪽으로만 답하는 모델)": [
            format_metric(baseline_metrics["정확도"]),
            format_metric(baseline_metrics["재현율"]),
            format_metric(baseline_metrics["정밀도"]),
            format_metric(baseline_metrics["F1"])
        ]
    },
    index=[
        "정확도",
        "재현율",
        "정밀도",
        "F1"
    ]
)

st.dataframe(
    metrics_table,
    use_container_width=True
)


# ==========================================
# 21. 평가 지표 설명
# ==========================================

with st.expander("평가 지표 계산 방법 확인하기"):

    st.markdown(
        """
        **정확도**

        전체 사람 중 실제 결과를 맞힌 사람의 비율입니다.

        **재현율**

        실제 뇌졸중인 사람 중 모델이 뇌졸중이라고 맞힌 비율입니다.

        분모는 실제 뇌졸중인 사람 수(TP + FN)입니다.

        **정밀도**

        뇌졸중이라고 예측한 사람 중 실제로 뇌졸중인 사람의 비율입니다.

        분모는 뇌졸중이라고 예측한 사람 수(TP + FP)입니다.

        **F1**

        정밀도와 재현율을 함께 고려한 지표입니다.

        정밀도와 재현율의 조화평균을 사용합니다.

        뇌졸중이라고 예측한 사람이 한 명도 없으면
        정밀도와 F1은 계산할 수 없습니다.
        """
    )


# ==========================================
# 22. 놓친 사람과 헛짚은 사람 분석
# ==========================================

st.header("6. 잘못 예측한 사람 확인하기")

st.write(
    """
    모델이 실제 뇌졸중인 사람을 놓쳤는지,
    뇌졸중이 아닌 사람을 뇌졸중이라고 잘못 예측했는지 살펴봅니다.
    """
)


# ==========================================
# 23. 모델 선택
# ==========================================

model_choice = st.radio(
    "확인할 모델 선택",
    options=[
        "로지스틱 회귀(확률로 답하는 모델)",
        "의사결정트리(질문으로 답하는 모델)"
    ],
    horizontal=True
)


if model_choice.startswith("로지스틱"):

    selected_predictions = logistic_test_pred

    selected_values = logistic_values

else:

    selected_predictions = tree_test_pred

    selected_values = tree_values


# ==========================================
# 24. 테스트 결과 데이터 만들기
# ==========================================

result_df = test_df.copy()

result_df["실제 결과"] = y_test.to_numpy()

result_df["예측 결과"] = selected_predictions

# 실제 뇌졸중인데 아니라고 예측한 사람
missed_people = result_df[
    (result_df["실제 결과"] == 1)
    & (result_df["예측 결과"] == 0)
].copy()

# 실제 뇌졸중이 아닌데 뇌졸중이라고 예측한 사람
false_alarm_people = result_df[
    (result_df["실제 결과"] == 0)
    & (result_df["예측 결과"] == 1)
].copy()


# ==========================================
# 25. 사람 목록 표시용 함수
# ==========================================

def prepare_people_table(data):

    display_df = data[
        [
            "id",
            "age",
            "avg_glucose_level",
            "hypertension",
            "heart_disease"
        ]
    ].copy()

    display_df = display_df.rename(
        columns={
            "id": "번호",
            "age": "나이",
            "avg_glucose_level": "평균 혈당",
            "hypertension": "고혈압 여부",
            "heart_disease": "심장병 여부"
        }
    )

    # 나이가 많은 순서로 정렬
    display_df = display_df.sort_values(
        "나이",
        ascending=False,
        na_position="last"
    )

    # 질환 여부를 우리말로 표시
    display_df["고혈압 여부"] = (
        display_df["고혈압 여부"]
        .map({
            0: "없음",
            1: "있음"
        })
    )

    display_df["심장병 여부"] = (
        display_df["심장병 여부"]
        .map({
            0: "없음",
            1: "있음"
        })
    )

    display_df = display_df.rename(
        columns={
            "번호": "번호",
            "나이": "나이",
            "평균 혈당": "평균 혈당",
            "고혈압 여부": "고혈압 여부",
            "심장병 여부": "심장병 여부"
        }
    )

    return display_df


# ==========================================
# 26. 두 목록을 나란히 표시
# ==========================================

left_col, right_col = st.columns(2)

with left_col:

    missed_count = len(missed_people)

    st.subheader(
        f"🚨 놓친 사람 ({missed_count:,}명)"
    )

    st.caption(
        "실제 뇌졸중인데 모델이 뇌졸중이 아니라고 예측한 사람 (FN)"
    )

    missed_table = prepare_people_table(
        missed_people
    )

    st.dataframe(
        missed_table.head(20),
        use_container_width=True,
        hide_index=True
    )

    if missed_count > 20:

        st.caption(
            f"전체 {missed_count:,}명 중 나이가 많은 "
            "앞 20명만 표시했습니다."
        )

    elif missed_count == 0:

        st.success(
            "놓친 사람이 없습니다."
        )


with right_col:

    false_alarm_count = len(false_alarm_people)

    st.subheader(
        f"⚠️ 헛짚은 사람 ({false_alarm_count:,}명)"
    )

    st.caption(
        "실제 뇌졸중이 아닌데 모델이 뇌졸중이라고 예측한 사람 (FP)"
    )

    false_alarm_table = prepare_people_table(
        false_alarm_people
    )

    st.dataframe(
        false_alarm_table.head(20),
        use_container_width=True,
        hide_index=True
    )

    if false_alarm_count > 20:

        st.caption(
            f"전체 {false_alarm_count:,}명 중 나이가 많은 "
            "앞 20명만 표시했습니다."
        )

    elif false_alarm_count == 0:

        st.success(
            "헛짚은 사람이 없습니다."
        )


# ==========================================
# 27. 혼동행렬과 목록 인원수 비교
# ==========================================

st.divider()

st.subheader("혼동행렬과 사람 목록 확인")

fn_count = selected_values["FN"]
fp_count = selected_values["FP"]

st.write(
    f"""
    **놓친 사람 목록: {missed_count:,}명 = 혼동행렬의 FN: {fn_count:,}명**

    **헛짚은 사람 목록: {false_alarm_count:,}명 = 혼동행렬의 FP: {fp_count:,}명**
    """
)

if missed_count == fn_count and false_alarm_count == fp_count:

    st.success(
        "두 목록의 인원수가 혼동행렬의 FN과 FP에 정확히 일치합니다."
    )

else:

    st.error(
        "목록과 혼동행렬의 인원수가 일치하지 않습니다."
    )


# ==========================================
# 28. 마무리
# ==========================================

st.divider()

st.subheader("실습에서 생각해 볼 점")

st.markdown(
    """
    1. 두 모델 중 재현율이 높은 모델은 무엇인가?
    2. 뇌졸중을 놓친 사람(FN)이 더 적은 모델은 무엇인가?
    3. 헛짚은 사람(FP)이 더 적은 모델은 무엇인가?
    4. 정확도가 높더라도 재현율이 낮을 수 있을까?
    5. 기준 모델과 비교했을 때 두 분류 모델은 어떤 차이가 있을까?
    """
)

st.caption(
    "※ 이 페이지는 교육용 데이터 분석 실습입니다. "
    "예측 결과는 실제 의료 진단이나 개인의 건강 상태를 판단하는 데 "
    "사용할 수 없습니다."
)
