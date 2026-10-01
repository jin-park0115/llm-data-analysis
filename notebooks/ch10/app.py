from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
BUNDLE_PATH = PROJECT_ROOT / "models" / "ch10_cancel_model_bundle.joblib"
CONTRACT_PATH = PROJECT_ROOT / "models" / "ch10_cancel_model_contract.json"

FEATURE_LABELS = {
    "age": "고객 나이",
    "order_month": "주문 월",
    "order_dayofweek": "주문 요일",
    "payment_method": "결제 수단",
    "gender": "성별",
    "city": "도시",
}
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
AGE_BINS = [0, 30, 40, 50, 60, 200]
AGE_BIN_LABELS = ["~29세", "30대", "40대", "50대", "60세~"]


# --- 1. 서빙 모델 로딩 (재학습 없이 저장된 Pipeline만 사용) ---
@st.cache_resource
def load_artifacts():
    for path in (BUNDLE_PATH, CONTRACT_PATH):
        if not path.is_file():
            raise FileNotFoundError(
                f"서빙 모델 파일이 없습니다: {path}\n"
                "ch10.ipynb의 STEP 15를 먼저 실행해 모델을 저장하세요."
            )
    bundle = joblib.load(BUNDLE_PATH)
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    return bundle, contract


def feature_columns(contract: dict) -> list[str]:
    return contract["numeric_features"] + contract["categorical_features"]


def cancel_probability(frame: pd.DataFrame, bundle: dict, contract: dict) -> np.ndarray:
    """저장된 Pipeline으로 취소(positive class) 확률을 계산한다."""
    model_input = frame[feature_columns(contract)].copy()
    for col in contract["numeric_features"]:
        model_input[col] = pd.to_numeric(model_input[col], errors="coerce")
    for col in contract["categorical_features"]:
        model_input[col] = model_input[col].where(
            model_input[col].isna(), model_input[col].astype(str)
        )

    pipeline = bundle["pipeline"]
    positive_index = list(pipeline.classes_).index(contract["positive_class"])
    return pipeline.predict_proba(model_input)[:, positive_index]


def predict(frame: pd.DataFrame, bundle: dict, contract: dict) -> pd.DataFrame:
    """취소 확률에 고정 Threshold를 적용해 예측 결과를 붙인다."""
    probs = cancel_probability(frame, bundle, contract)
    result = frame.copy()
    result["cancel_probability"] = probs.round(4)
    result["predicted_is_cancelled"] = (probs >= bundle["threshold"]).astype(int)
    return result


@st.cache_data
def baseline_values(_bundle: dict, contract: dict) -> dict:
    """Pipeline 안의 Imputer가 Train 세트에서 학습한 중앙값/최빈값을 기준값으로 사용한다."""
    pre = _bundle["pipeline"].named_steps["preprocessor"]
    baseline = {}
    for name, cols in (("num", contract["numeric_features"]), ("cat", contract["categorical_features"])):
        stats = pre.named_transformers_[name].named_steps["imputer"].statistics_
        baseline.update(dict(zip(cols, stats)))
    return baseline


@st.cache_data
def feature_importance(_bundle: dict, contract: dict) -> pd.DataFrame:
    """One-Hot으로 펼쳐진 중요도를 원래 입력 Feature 단위로 합산한다."""
    pre = _bundle["pipeline"].named_steps["preprocessor"]
    clf = _bundle["pipeline"].named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        scores = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        scores = np.abs(clf.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    names = pre.get_feature_names_out()
    totals = dict.fromkeys(feature_columns(contract), 0.0)
    for name, score in zip(names, scores):
        raw = name.split("__", 1)[1]
        # 긴 이름부터 매칭해 접두어가 겹치는 Feature를 구분한다
        for col in sorted(totals, key=len, reverse=True):
            if raw == col or raw.startswith(col + "_"):
                totals[col] += float(score)
                break

    total = sum(totals.values()) or 1.0
    return pd.DataFrame(
        [{"feature": FEATURE_LABELS.get(c, c), "importance": v / total} for c, v in totals.items()]
    ).sort_values("importance", ascending=False)


def explain_prediction(raw: pd.DataFrame, bundle: dict, contract: dict) -> pd.DataFrame:
    """각 Feature를 기준값으로 바꿨을 때 취소 확률이 얼마나 달라지는지 계산한다."""
    baseline = baseline_values(bundle, contract)
    cols = feature_columns(contract)
    variants = [raw.iloc[0].to_dict()]
    for col in cols:
        row = raw.iloc[0].to_dict()
        row[col] = baseline[col]
        variants.append(row)

    probs = cancel_probability(pd.DataFrame(variants), bundle, contract)
    return pd.DataFrame(
        [
            {
                "feature": FEATURE_LABELS.get(col, col),
                "입력값": str(raw.iloc[0][col]),
                "기준값": str(baseline[col]),
                "영향": float(probs[0] - probs[i + 1]),
            }
            for i, col in enumerate(cols)
        ]
    )


@st.cache_data
def validation_confusion(contract: dict) -> pd.DataFrame:
    """저장된 Validation Precision/Recall 표에서 Threshold별 TP/FP/FN 건수를 역산한다."""
    th_df = pd.DataFrame(contract["validation_threshold_metrics"])
    n_val = contract["split_sizes"]["validation"]

    # Recall = TP / P 이므로 모든 Recall 값을 정수 TP로 만드는 최소 P가 Validation 취소 건수다
    recalls = th_df["Recall"].to_numpy()
    positives = next(
        p for p in range(1, n_val + 1) if np.allclose(recalls * p, np.round(recalls * p), atol=0.01)
    )

    rows = []
    for _, r in th_df.iterrows():
        tp = int(round(r["Recall"] * positives))
        # Precision=0이고 TP=0이면 예측 양성 건수를 알 수 없으므로 FP는 미확정 처리
        fp = int(round(tp / r["Precision"])) - tp if r["Precision"] > 0 else (None if tp == 0 else 0)
        rows.append({"Threshold": r["Threshold"], "TP": tp, "FP": fp, "FN": positives - tp})
    result = pd.DataFrame(rows)
    result.attrs["positives"] = positives
    result.attrs["negatives"] = n_val - positives
    return result


st.set_page_config(page_title="주문 취소 예측", page_icon=":material/local_shipping:", layout="wide")

st.title(":material/local_shipping: 주문 취소 예측 대시보드")
st.caption(
    "Chapter 10 노트북에서 Validation으로 고정한 모델과 Threshold를 그대로 불러와 예측합니다. "
    "예측 결과는 취소 위험 신호일 뿐, 취소 원인을 의미하지 않습니다."
)

try:
    bundle, contract = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.stop()

threshold = bundle["threshold"]
options = contract["category_options"]
age_range = contract["numeric_ranges"]["age"]

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.subheader(":material/model_training: 서빙 모델")
    st.markdown(f"**모델**: {bundle['model_name']}")
    st.markdown(f"**Threshold**: {threshold}")
    st.markdown(f"**예측 시점**: {contract['prediction_time']}")
    st.markdown("**입력 Feature**")
    st.markdown("\n".join(f"- `{c}` {FEATURE_LABELS.get(c, '')}" for c in feature_columns(contract)))
    st.caption("Threshold는 Validation에서 고정되었으며, Test 결과를 보고 다시 조정하지 않습니다.")

tab_dashboard, tab_single, tab_batch, tab_cost = st.tabs(
    [
        ":material/dashboard: 모델 성능",
        ":material/person_search: 단건 예측",
        ":material/upload_file: 일괄 예측",
        ":material/payments: 비용 시뮬레이터",
    ]
)

# --- 2. 모델 성능 대시보드 ---
with tab_dashboard:
    m = contract["test_metrics"]
    cm = contract["test_confusion_matrix"]
    dummy = next(r for r in contract["validation_model_comparison"] if r["Model"] == "Dummy")

    st.subheader("Final Test 성적표")
    with st.container(horizontal=True):
        st.metric("Recall (취소 탐지율)", f"{m['recall']:.1%}", border=True,
                  help="실제 취소 주문 중 모델이 잡아낸 비율")
        st.metric("Precision", f"{m['precision']:.1%}", border=True,
                  help="취소 위험으로 예측한 주문 중 실제 취소 비율")
        st.metric("F1-Score", f"{m['f1']:.4f}", f"Dummy 대비 +{m['f1'] - dummy['F1']:.4f}", border=True)
        st.metric("Accuracy", f"{m['accuracy']:.1%}", border=True,
                  help="클래스 불균형 때문에 단독 지표로 쓰기 어렵습니다.")

    threshold_rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(
        strokeDash=[4, 4], color="red"
    ).encode(x="t:Q")

    col_cm, col_prob = st.columns(2)
    with col_cm:
        with st.container(border=True):
            st.markdown("**Confusion matrix (Test)**")
            cm_df = pd.DataFrame(
                [
                    {"실제": "완료(0)", "예측": "완료(0)", "건수": cm["tn"], "구분": "TN"},
                    {"실제": "완료(0)", "예측": "취소(1)", "건수": cm["fp"], "구분": "FP 헛경보"},
                    {"실제": "취소(1)", "예측": "완료(0)", "건수": cm["fn"], "구분": "FN 놓친 취소"},
                    {"실제": "취소(1)", "예측": "취소(1)", "건수": cm["tp"], "구분": "TP"},
                ]
            )
            base = alt.Chart(cm_df).encode(
                x=alt.X("예측:N", sort=["완료(0)", "취소(1)"]),
                y=alt.Y("실제:N", sort=["완료(0)", "취소(1)"]),
            )
            heat = base.mark_rect().encode(
                color=alt.Color("건수:Q", scale=alt.Scale(scheme="blues"), legend=None),
                tooltip=["구분", "건수"],
            )
            text = base.mark_text(fontSize=16).encode(
                text=alt.Text("label:N"),
                color=alt.condition("datum.건수 > 12", alt.value("white"), alt.value("black")),
            ).transform_calculate(label="datum.구분 + ': ' + datum.건수")
            st.altair_chart(heat + text, height=260)

    with col_prob:
        with st.container(border=True):
            st.markdown("**Test 세트 취소 확률 분포**")
            prob_df = pd.DataFrame(contract["test_probabilities"])
            prob_df["실제"] = prob_df["actual"].map({0: "완료(0)", 1: "취소(1)"})
            hist = alt.Chart(prob_df).mark_bar(opacity=0.7).encode(
                x=alt.X("probability:Q", bin=alt.Bin(step=0.05), title="취소 확률"),
                y=alt.Y("count():Q", title="주문 수").stack(None),
                color=alt.Color("실제:N", legend=alt.Legend(orient="top")),
            )
            st.altair_chart(hist + threshold_rule, height=260)

    col_models, col_th = st.columns(2)
    with col_models:
        with st.container(border=True):
            st.markdown("**Validation 모델 비교**")
            st.caption("F1 → Recall → Precision 순으로 Dummy를 제외한 후보 중 선택")
            model_df = pd.DataFrame(contract["validation_model_comparison"])
            st.dataframe(
                model_df,
                hide_index=True,
                column_config={
                    c: st.column_config.ProgressColumn(c, format="%.3f", min_value=0, max_value=1)
                    for c in ["Accuracy", "Precision", "Recall", "F1"]
                },
            )

    with col_th:
        with st.container(border=True):
            st.markdown("**Validation Threshold 탐색**")
            th_long = pd.DataFrame(contract["validation_threshold_metrics"]).melt(
                id_vars="Threshold", var_name="지표", value_name="값"
            )
            lines = alt.Chart(th_long).mark_line(point=True).encode(
                x=alt.X("Threshold:Q"),
                y=alt.Y("값:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("지표:N", legend=alt.Legend(orient="top")),
                tooltip=["Threshold", "지표", "값"],
            )
            st.altair_chart(lines + threshold_rule, height=260)

    with st.container(border=True):
        st.markdown("**Feature 중요도**")
        st.caption("One-Hot 인코딩된 컬럼의 중요도를 원래 입력 Feature 단위로 합산한 값입니다. 중요도는 인과관계를 뜻하지 않습니다.")
        imp_df = feature_importance(bundle, contract)
        imp_chart = alt.Chart(imp_df).mark_bar().encode(
            x=alt.X("importance:Q", title="상대 중요도", axis=alt.Axis(format="%")),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature", alt.Tooltip("importance:Q", format=".1%")],
        )
        st.altair_chart(imp_chart, height=220)

    counts = contract["class_counts"]
    sizes = contract["split_sizes"]
    st.info(
        f"전체 {sum(counts.values())}건 중 취소 {counts['1']}건 "
        f"({counts['1'] / sum(counts.values()):.1%}) · "
        f"Train {sizes['train']} / Validation {sizes['validation']} / Test {sizes['test']} (Stratified Random Split). "
        "소규모 데이터와 무작위 분할의 한계로, 실운영 전 시계열(Out-of-Time) 검증이 추가로 필요합니다.",
        icon=":material/info:",
    )

# --- 3. 단건 예측 ---
with tab_single:
    st.subheader("신규 주문 취소 위험 예측")
    with st.form("order_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            payment_method = st.selectbox(FEATURE_LABELS["payment_method"], options["payment_method"])
            order_month = st.selectbox(
                FEATURE_LABELS["order_month"], options["order_month"],
                index=len(options["order_month"]) - 1,
            )
        with c2:
            days = sorted(options["order_dayofweek"], key=lambda d: DAY_ORDER.index(d) if d in DAY_ORDER else 99)
            order_dayofweek = st.selectbox(FEATURE_LABELS["order_dayofweek"], days)
            city = st.selectbox(FEATURE_LABELS["city"], options["city"])
        with c3:
            gender = st.segmented_control(FEATURE_LABELS["gender"], options["gender"], default=options["gender"][0])
            age = st.number_input(
                FEATURE_LABELS["age"], min_value=0, max_value=100,
                value=int(age_range["median"]), step=1,
            )
        submitted = st.form_submit_button("예측하기", icon=":material/insights:", type="primary")

    if submitted:
        raw = pd.DataFrame([{
            "age": age,
            "order_month": order_month,
            "order_dayofweek": order_dayofweek,
            "payment_method": payment_method,
            "gender": gender,
            "city": city,
        }])
        try:
            row = predict(raw, bundle, contract).iloc[0]
            prob = float(row["cancel_probability"])
            st.session_state.history.append(
                {**raw.iloc[0].to_dict(), "cancel_probability": prob,
                 "predicted_is_cancelled": int(row["predicted_is_cancelled"])}
            )

            col_result, col_explain = st.columns(2)
            with col_result:
                with st.container(border=True):
                    if row["predicted_is_cancelled"] == 1:
                        st.error("**취소 위험 주문**: 출고 전 주문 확인 알림 발송을 권장합니다.", icon=":material/warning:")
                    else:
                        st.success("**정상 주문 예상**: 일반 출고 프로세스로 진행합니다.", icon=":material/check_circle:")
                    with st.container(horizontal=True):
                        st.metric("취소 확률 (모델 출력)", f"{prob:.1%}", border=True)
                        st.metric("판정 기준 Threshold", f"{threshold:.0%}", f"{prob - threshold:+.1%}p",
                                  delta_color="inverse", border=True)
                    st.progress(min(prob, 1.0), text=f"취소 확률 {prob:.1%} / 기준 {threshold:.0%}")

            with col_explain:
                with st.container(border=True):
                    st.markdown("**예측 근거**")
                    st.caption(
                        "각 입력을 Train 기준값(중앙값/최빈값)으로 바꿨을 때보다 취소 확률이 얼마나 높거나 낮은지 보여줍니다. "
                        "모델의 판단 경향일 뿐, 취소 원인이 아닙니다."
                    )
                    exp_df = explain_prediction(raw, bundle, contract)
                    exp_df["방향"] = np.where(exp_df["영향"] >= 0, "확률 증가", "확률 감소")
                    exp_chart = alt.Chart(exp_df).mark_bar().encode(
                        x=alt.X("영향:Q", title="취소 확률 변화", axis=alt.Axis(format="+%")),
                        y=alt.Y("feature:N", sort=alt.EncodingSortField("영향", op="sum", order="descending"), title=None),
                        color=alt.Color(
                            "방향:N",
                            scale=alt.Scale(domain=["확률 증가", "확률 감소"], range=["#d62728", "#1f77b4"]),
                            legend=alt.Legend(orient="top", title=None),
                        ),
                        tooltip=["feature", "입력값", "기준값", alt.Tooltip("영향:Q", format="+.1%")],
                    )
                    st.altair_chart(exp_chart, height=240)
        except Exception as exc:
            st.error(f"예측 처리 중 오류가 발생했습니다: {exc}")

    if st.session_state.history:
        with st.container(border=True):
            st.markdown(f"**이번 세션 예측 기록** ({len(st.session_state.history)}건)")
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(
                history_df.iloc[::-1],
                hide_index=True,
                column_config={
                    "cancel_probability": st.column_config.ProgressColumn(
                        "취소 확률", format="%.3f", min_value=0, max_value=1
                    ),
                    "predicted_is_cancelled": st.column_config.CheckboxColumn("취소 위험"),
                },
            )
            with st.container(horizontal=True):
                st.download_button(
                    "기록 CSV 받기",
                    history_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="ch10_single_prediction_history.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )
                if st.button("기록 지우기", icon=":material/delete:"):
                    st.session_state.history = []
                    st.rerun()

# --- 4. 일괄 예측 ---
with tab_batch:
    st.subheader("CSV 일괄 예측")
    cols = feature_columns(contract)
    st.markdown("필수 컬럼: " + ", ".join(f"`{c}`" for c in cols))

    template = pd.DataFrame([{
        "age": int(age_range["median"]),
        "order_month": options["order_month"][-1],
        "order_dayofweek": "Monday",
        "payment_method": options["payment_method"][0],
        "gender": options["gender"][0],
        "city": "서울" if "서울" in options["city"] else options["city"][0],
    }])[cols]
    st.download_button(
        "입력 템플릿 CSV 받기",
        template.to_csv(index=False).encode("utf-8-sig"),
        file_name="ch10_cancel_input_template.csv",
        mime="text/csv",
        icon=":material/download:",
    )

    uploaded = st.file_uploader("주문 CSV 업로드", type="csv")
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded, encoding="utf-8-sig")
        except Exception as exc:
            st.error(f"CSV를 읽을 수 없습니다: {exc}")
            st.stop()

        missing = [c for c in cols if c not in batch.columns]
        if missing:
            st.error(f"필수 컬럼이 없습니다: {missing}")
        else:
            unseen = {
                c: sorted(set(batch[c].dropna().astype(str)) - set(options[c]))
                for c in contract["categorical_features"]
            }
            unseen = {c: v for c, v in unseen.items() if v}
            if unseen:
                st.warning(
                    "학습에 없던 범주가 포함되어 있습니다 (해당 값은 모델에서 무시됩니다): "
                    + "; ".join(f"{c}={v}" for c, v in unseen.items()),
                    icon=":material/warning:",
                )

            result = predict(batch, bundle, contract)
            flagged = int(result["predicted_is_cancelled"].sum())
            with st.container(horizontal=True):
                st.metric("업로드 주문 수", f"{len(result)}건", border=True)
                st.metric("취소 위험 주문", f"{flagged}건", border=True)
                st.metric("취소 위험 비율", f"{flagged / len(result):.1%}", border=True)

            with st.container(border=True):
                st.markdown("**세그먼트별 취소 위험**")
                segment_labels = {"age_band": "연령대", **{c: FEATURE_LABELS[c] for c in contract["categorical_features"]}}
                segment = st.segmented_control(
                    "세그먼트 기준", list(segment_labels), default="payment_method",
                    format_func=segment_labels.get, key="segment",
                ) or "payment_method"
                seg_frame = result.copy()
                seg_frame["age_band"] = pd.cut(
                    pd.to_numeric(seg_frame["age"], errors="coerce"),
                    bins=AGE_BINS, labels=AGE_BIN_LABELS, right=False,
                ).astype(str)
                seg_frame[segment] = seg_frame[segment].fillna("(결측)").astype(str)
                seg_df = (
                    seg_frame.groupby(segment)
                    .agg(주문수=("predicted_is_cancelled", "size"),
                         취소위험비율=("predicted_is_cancelled", "mean"),
                         평균취소확률=("cancel_probability", "mean"))
                    .reset_index()
                )
                overall = result["predicted_is_cancelled"].mean()
                seg_bar = alt.Chart(seg_df).mark_bar().encode(
                    x=alt.X(f"{segment}:N", title=segment_labels[segment], sort="-y"),
                    y=alt.Y("취소위험비율:Q", title="취소 위험 비율", axis=alt.Axis(format="%")),
                    tooltip=[segment, "주문수", alt.Tooltip("취소위험비율:Q", format=".1%"),
                             alt.Tooltip("평균취소확률:Q", format=".3f")],
                )
                avg_rule = alt.Chart(pd.DataFrame({"y": [overall]})).mark_rule(
                    strokeDash=[4, 4], color="gray"
                ).encode(y="y:Q")
                st.altair_chart(seg_bar + avg_rule, height=280)
                st.caption(
                    f"회색 점선은 전체 평균({overall:.1%})입니다. 특정 집단에만 경보가 몰리면 "
                    "고객 경험·공정성 측면에서 추가 검토가 필요합니다. 주문 수가 적은 세그먼트는 해석에 주의하세요."
                )

            st.dataframe(
                result.sort_values("cancel_probability", ascending=False),
                hide_index=True,
                column_config={
                    "cancel_probability": st.column_config.ProgressColumn(
                        "취소 확률", format="%.3f", min_value=0, max_value=1
                    ),
                    "predicted_is_cancelled": st.column_config.CheckboxColumn("취소 위험"),
                },
            )
            st.download_button(
                "예측 결과 CSV 받기",
                result.to_csv(index=False).encode("utf-8-sig"),
                file_name="ch10_cancel_predictions.csv",
                mime="text/csv",
                icon=":material/download:",
            )

# --- 5. 비용 시뮬레이터 ---
with tab_cost:
    st.subheader("Threshold별 예상 비용 (What-if)")
    st.caption(
        "Validation 세트의 Threshold 탐색 결과로 TP/FP/FN 건수를 계산하고, 입력한 비용 단가로 총비용을 비교합니다. "
        "서빙 Threshold는 바뀌지 않으며, 변경하려면 노트북에서 Validation 규칙을 다시 정하고 재학습해야 합니다."
    )

    with st.container(horizontal=True):
        fp_cost = st.number_input("FP 비용 (헛경보 1건, 원)", min_value=0, value=50, step=10,
                                  help="정상 고객에게 확인 알림 문자를 보내는 비용")
        fn_cost = st.number_input("FN 비용 (놓친 취소 1건, 원)", min_value=0, value=6000, step=500,
                                  help="출고 후 취소로 발생하는 왕복 택배비·포장비·인건비")
        orders_per_month = st.number_input("월 주문 수 (환산용)", min_value=1, value=10000, step=1000)

    conf = validation_confusion(contract)
    n_val = contract["split_sizes"]["validation"]
    known = conf.dropna(subset=["FP"]).copy()
    known["FP"] = known["FP"].astype(int)
    known["총비용"] = known["FP"] * fp_cost + known["FN"] * fn_cost
    known["월 환산 비용"] = known["총비용"] / n_val * orders_per_month

    no_model_cost = conf.attrs["positives"] * fn_cost
    best = known.sort_values(["총비용", "Threshold"], ascending=[True, False]).iloc[0]
    serving = known[np.isclose(known["Threshold"], threshold)]

    with st.container(horizontal=True):
        if not serving.empty:
            s = serving.iloc[0]
            st.metric("서빙 Threshold 비용", f"{s['월 환산 비용']:,.0f}원/월",
                      help=f"Threshold {threshold} · FP {s['FP']}건 · FN {s['FN']}건 (Validation)", border=True)
        st.metric(
            "비용 최소 Threshold", f"{best['Threshold']:.2f}",
            f"{best['월 환산 비용']:,.0f}원/월", delta_color="off", border=True,
            help=f"FP {best['FP']}건 · FN {best['FN']}건 (Validation)",
        )
        st.metric("모델 미사용 비용", f"{no_model_cost / n_val * orders_per_month:,.0f}원/월",
                  help="알림을 전혀 보내지 않아 모든 취소를 놓치는 경우", border=True)

    with st.container(border=True):
        cost_line = alt.Chart(known).mark_line(point=True).encode(
            x=alt.X("Threshold:Q"),
            y=alt.Y("월 환산 비용:Q", title="월 환산 예상 비용 (원)"),
            tooltip=["Threshold", "TP", "FP", "FN", alt.Tooltip("월 환산 비용:Q", format=",.0f")],
        )
        best_point = alt.Chart(known[known["Threshold"] == best["Threshold"]]).mark_point(
            size=150, color="green", filled=True
        ).encode(x="Threshold:Q", y="월 환산 비용:Q")
        st.altair_chart(cost_line + best_point + threshold_rule, height=300)
        st.caption("빨간 점선: 서빙 Threshold · 초록 점: 입력한 비용 기준 최소 비용 지점")

    dropped = conf[conf["FP"].isna()]["Threshold"].tolist()
    if dropped:
        st.caption(
            f"Threshold {min(dropped):.2f} 이상은 Validation에서 취소를 1건도 잡지 못해 "
            "FP 건수를 역산할 수 없으므로 그래프에서 제외했습니다."
        )
    with st.expander("Threshold별 상세 건수 (Validation)"):
        st.dataframe(
            known[["Threshold", "TP", "FP", "FN", "총비용", "월 환산 비용"]],
            hide_index=True,
            column_config={
                "총비용": st.column_config.NumberColumn(format="localized"),
                "월 환산 비용": st.column_config.NumberColumn(format="localized"),
            },
        )
