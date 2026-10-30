
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split

# ==========================================
# 기본 설정
# ==========================================
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 뇌졸중 예측 실습실")
st.caption("데이터를 이해하고 뇌졸중 예측 모델을 만들어 보는 실습 공간입니다.")

DATA_URL = "https://raw.githubusercontent.com/Lin-1120/data/refs/heads/main/stroke.csv"

# JSON 저장 위치
RECORD_DIR = Path(__file__).resolve().parent / "내_기록"
RECORD_FILE = RECORD_DIR / "뇌졸중_데이터_설명.json"

EXPECTED_COLUMNS = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke"
]


# ==========================================
# JSON 파일 불러오기
# ==========================================
def load_notes():

    if not RECORD_FILE.exists():
        return {}, ""

    try:
        with open(RECORD_FILE, "r", encoding="utf-8") as file:
            saved = json.load(file)

        return (
            saved.get("meanings", {}),
            saved.get("source", "")
        )

    except (json.JSONDecodeError, OSError):
        return {}, ""


# ==========================================
# JSON 파일 저장하기
# ==========================================
def save_notes(meanings, source):

    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "meanings": meanings,
        "source": source
    }

    with open(RECORD_FILE, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )


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
    st.write("인터넷 연결이나 CSV 주소를 확인해 주세요.")

    with st.expander("오류 내용 확인"):
        st.code(str(error))

    st.stop()


# ==========================================
# 열 확인
# ==========================================
missing_columns = [
    col for col in EXPECTED_COLUMNS
    if col not in df.columns
]

if missing_columns:

    st.error(
        "CSV에서 다음 열을 찾을 수 없습니다: "
        + ", ".join(missing_columns)
    )

    st.stop()


# ==========================================
# 데이터 소개
# ==========================================
st.header("1. 데이터 소개")

st.write(
    """
    이 데이터는 여러 사람의 기본 정보와 건강 관련 정보를 담고 있습니다.

    각 행은 한 사람의 기록이며,
    나이, 고혈압 여부, 심장병 여부, 평균 혈당 수치,
    BMI, 흡연 상태 등의 정보를 포함합니다.

    마지막 열인 `stroke`는 뇌졸중 발생 여부를 나타냅니다.
    """
)


# ==========================================
# 주요 데이터 요약
# ==========================================
total_people = len(df)
column_count = len(df.columns)

stroke_count = (
    pd.to_numeric(df["stroke"], errors="coerce") == 1
).sum()

stroke_ratio = (
    stroke_count / total_people * 100
    if total_people > 0
    else 0
)

st.subheader("데이터 한눈에 보기")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "전체 사람 수",
    f"{total_people:,}명"
)

col2.metric(
    "열 개수",
    f"{column_count:,}개"
)

col3.metric(
    "stroke가 1인 사람 수",
    f"{stroke_count:,}명"
)

col4.metric(
    "stroke가 1인 비율",
    f"{stroke_ratio:.2f}%"
)


# ==========================================
# 열 이름과 뜻 정리
# ==========================================
st.header("2. 열 이름과 데이터 살펴보기")

st.write(
    "교재를 참고하여 각 열의 우리말 뜻을 직접 입력해 보세요."
)

saved_meanings, saved_source = load_notes()

summary_rows = []

for column in df.columns:

    series = df[column]

    if pd.api.types.is_numeric_dtype(series):

        value_kind = (
            f"숫자형 · 고유값 "
            f"{series.nunique(dropna=True):,}종"
        )

    else:

        value_kind = (
            f"문자형 · 고유값 "
            f"{series.nunique(dropna=True):,}종"
        )

    summary_rows.append({
        "열 이름": column,
        "우리말 뜻": saved_meanings.get(column, ""),
        "값의 종류": value_kind,
        "빈 값 개수": int(series.isna().sum())
    })


summary_df = pd.DataFrame(summary_rows)

edited_df = st.data_editor(
    summary_df,
    use_container_width=True,
    hide_index=True,
    disabled=[
        "열 이름",
        "값의 종류",
        "빈 값 개수"
    ],
    column_config={
        "열 이름": st.column_config.TextColumn(
            "열 이름"
        ),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 참고하여 직접 입력하세요."
        ),
        "값의 종류": st.column_config.TextColumn(
            "값의 종류"
        ),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수",
            format="%d"
        )
    },
    key="meaning_editor"
)

st.caption(
    "값의 종류는 숫자형 또는 문자형 여부와 "
    "결측치를 제외한 고유값 개수를 나타냅니다."
)


# ==========================================
# 원본 데이터 처음 5줄
# ==========================================
st.header("3. 원본 데이터 확인")

st.write("데이터의 처음 다섯 줄입니다.")

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 데이터 출처 기록
# ==========================================
st.header("4. 데이터 출처 기록")

source_text = st.text_area(
    "교재에 나온 데이터 출처를 입력하세요.",
    value=saved_source,
    placeholder="데이터 출처를 여기에 입력하세요.",
    height=100
)


# ==========================================
# 저장 버튼
# ==========================================
if st.button(
    "💾 우리말 뜻과 출처 저장",
    type="primary",
    use_container_width=True
):

    meanings_to_save = {}

    for _, row in edited_df.iterrows():

        meaning = row["우리말 뜻"]

        if pd.isna(meaning):
            meaning = ""

        meanings_to_save[str(row["열 이름"])] = str(meaning)

    try:

        save_notes(
            meanings_to_save,
            source_text
        )

        st.success(
            "우리말 뜻과 데이터 출처를 저장했습니다!"
        )

        st.caption(
            f"저장 위치: {RECORD_FILE}"
        )

    except OSError as error:

        st.error("파일 저장에 실패했습니다.")

        st.code(str(error))


st.caption(
    "저장 파일: 내_기록/뇌졸중_데이터_설명.json"
)

st.info(
    "새로고침하면 저장된 내용을 다시 불러옵니다. "
    "단, Streamlit Cloud의 로컬 파일은 재배포나 "
    "서버 재시작 시 초기화될 수 있습니다."
)
