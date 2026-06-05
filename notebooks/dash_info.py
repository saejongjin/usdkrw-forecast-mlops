from dash import Dash, html, dcc, Input, Output, State, MATCH
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import math
from dash import ctx, no_update
from dash.exceptions import PreventUpdate

# =========================================================
# 데이터 및 모델 불러오기
# =========================================================
DATA_PATH = "/app/data/processed/market_data_1m_24h_interpolated.csv"
MODEL_PATH = "/app/data/processed/predict_model.pkl"
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)


# 모델 입력 변수 자동 추출
if hasattr(model, "feature_names_in_"):
    feature_cols = list(model.feature_names_in_)
else:
    # feature_names_in_이 없는 경우를 대비한 fallback
    feature_cols = [c for c in df.columns if c != "timestamp"]

# CSV에 실제 존재하는 변수만 사용
feature_cols = [c for c in feature_cols if c in df.columns]

# 변수 중요도 자동 추출
if hasattr(model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_[:len(feature_cols)]
    }).sort_values("importance", ascending=False)
else:
    # 중요도가 없는 모델이면 원래 변수 순서 유지
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": np.nan
    })

ordered_features = importance_df["feature"].tolist()


def decimals_from_std(std_value, sig=2):
    """
    표준편차 기준으로 표시할 소수점 자리수 결정
    유효숫자 2개 정도 유지
    """
    if pd.isna(std_value) or std_value <= 0:
        return 2

    order = math.floor(math.log10(abs(std_value)))
    decimals = max(0, sig - 1 - order)
    return decimals


def round_by_std(x, std_value, sig=2):
    decimals = decimals_from_std(std_value, sig=sig)
    return round(float(x), decimals)


def format_by_std(x, std_value, sig=2):
    decimals = decimals_from_std(std_value, sig=sig)
    return f"{float(x):,.{decimals}f}"


feature_meta = {}

for col in ordered_features:
    std_value = float(df[col].std())
    decimals = decimals_from_std(std_value, sig=2)

    min_val_raw = float(df[col].min())
    max_val_raw = float(df[col].max())
    latest_val_raw = float(df[col].iloc[-1])

    min_val = round_by_std(min_val_raw, std_value, sig=2)
    max_val = round_by_std(max_val_raw, std_value, sig=2)
    latest_val = round_by_std(latest_val_raw, std_value, sig=2)
    mid_val = round_by_std((min_val_raw + max_val_raw) / 2, std_value, sig=2)

    # step도 같은 precision 기준으로 깔끔하게
    step = 10 ** (-decimals)

    feature_meta[col] = {
        "std": std_value,
        "decimals": decimals,
        "min": min_val,
        "max": max_val,
        "mid": mid_val,
        "latest": latest_val,
        "step": step
    }

# timestamp 처리
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# 학습 데이터 범위
if "timestamp" in df.columns:
    ksj1 = df["timestamp"].min().strftime("%Y-%m-%d %H:%M")
    ksj2 = df["timestamp"].max().strftime("%Y-%m-%d %H:%M")
else:
    ksj1 = "-"
    ksj2 = "-"


# =========================================================
# Dash App
# =========================================================
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.BOOTSTRAP
    ]
)


# =========================
# 열 너비 옵션
# =========================
row1_cols = "1.1fr 2fr 1.1fr"   # 1행: 좌 / 중 / 우
row2_cols = "1.1fr 2fr 1.1fr"   # 2행: 좌 / 중 / 우

row1_height = "250px"
row2_height = "380px"


# =========================
# 공통 카드 생성 함수
# =========================
def make_card(title, number=None, height="260px"):
    return html.Div(
        className="card dashboard-card",
        style={"minHeight": height},
        children=[
            html.Div(
                className="card-title-row",
                children=[
                    html.Span(str(number), className="card-number") if number else None,
                    html.H3(title, className="card-title")
                ]
            ),

            # 내용은 비워둠
            html.Div(className="card-body-empty")
        ]
    )

# =========================
# 모델 요약 항목 생성 함수
# =========================

def summary_item(icon_class, label, value):
    return html.Div(
        className="summary-item-modern",
        children=[
            html.Div(
                className="summary-icon-box",
                children=html.I(className=f"bi {icon_class} summary-icon")
            ),
            html.Div(
                className="summary-text-box",
                children=[
                    html.Div(label, className="summary-label-modern"),
                    html.Div(value, className="summary-value-modern")
                ]
            )
        ]
    )

# =========================
# 모델 요약 카드
# =========================

def make_model_summary_card(height="260px"):
    now_time = datetime.now()
    pred_time = now_time + timedelta(minutes=15)

    now_str = now_time.strftime("%Y-%m-%d %H:%M:%S")
    pred_str = pred_time.strftime("%Y-%m-%d %H:%M:%S")

    return html.Div(
        className="card model-summary-card",
        style={"minHeight": height},
        children=[
            html.Div(
                className="card-title-row",
                children=[
                    html.Span("3", className="card-number"),
                    html.H3("모델 요약", className="card-title")
                ]
            ),

            html.Div(
                className="model-summary-modern",
                children=[
                    summary_item(
                        "bi-cpu",
                        "모델명",
                        "RandomForest Classifier"
                    ),
                    summary_item(
                        "bi-clock",
                        "예측 시점",
                        f"{pred_str} | 15분 후"
                    ),
                    summary_item(
                        "bi-database",
                        "최신 데이터",
                        now_str
                    ),
                    summary_item(
                        "bi-calendar3",
                        "학습 데이터 범위",
                        f"{ksj1} ~ {ksj2}"
                    ),
                ]
            )
        ]
    )

# =========================================================
# Gauge Chart 생성 함수
# =========================================================
def make_probability_gauge(prob_1):
    prob_percent = prob_1 * 100

    return go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=prob_percent,
            number={
                "suffix": "%",
                "font": {"size": 34}
            },
            delta={
                "reference": 50,
                "increasing": {"color": "#16a34a"},
                "decreasing": {"color": "#dc2626"}
            },
            title={
                "text": "Class 1 확률",
                "font": {"size": 16}
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#9ca3af"
                },
                "bar": {"color": "#111827"},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#e5e7eb",
                "steps": [
                    {"range": [0, 40], "color": "#ef4444"},
                    {"range": [40, 60], "color": "#facc15"},
                    {"range": [60, 100], "color": "#22c55e"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 4},
                    "thickness": 0.8,
                    "value": 50
                }
            }
        )
    ).update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="white",
        font={"family": "Arial"}
    )


# =========================================================
# 시뮬레이션 입력 행
# =========================================================
def make_simulation_row(feature):
    meta = feature_meta[feature]

    min_val = meta["min"]
    max_val = meta["max"]
    mid_val = meta["mid"]
    latest_val = meta["latest"]
    step = meta["step"]
    std_value = meta["std"]

    return html.Div(
        className="sim-row",
        children=[
            # 1열: 변수명
            html.Div(
                feature,
                className="sim-var-name"
            ),

            # 2열: 조정바
            dcc.Slider(
                id={"type": "sim-slider", "feature": feature},
                min=min_val,
                max=max_val,
                step=step,
                value=latest_val,
                marks={
                    min_val: format_by_std(min_val, std_value),
                    mid_val: format_by_std(mid_val, std_value),
                    max_val: format_by_std(max_val, std_value),
                },
                tooltip=None,
                className="sim-slider"
            ),

            # 3열: 현재값 표시
            html.Div(
                id={"type": "sim-value-label", "feature": feature},
                className="sim-value-label",
                children=format_by_std(latest_val, std_value)
            ),

            # 4열: +/- 버튼
            html.Div(
                className="sim-step-buttons",
                children=[
                    html.Button(
                        "−",
                        id={"type": "sim-minus", "feature": feature},
                        n_clicks=0,
                        className="sim-step-button"
                    ),
                    html.Button(
                        "+",
                        id={"type": "sim-plus", "feature": feature},
                        n_clicks=0,
                        className="sim-step-button"
                    ),
                ]
            )
        ]
    )

# =========================================================
# 5번 시뮬레이션 패널
# =========================================================
def make_simulation_panel_card(height="360px"):
    return html.Div(
        className="card dashboard-card simulation-card",
        style={"minHeight": height},
        children=[
            html.Div(
                className="card-title-row",
                children=[
                    html.Span("5", className="card-number"),
                    html.H3("시뮬레이션 패널", className="card-title")
                ]
            ),

            html.Div(
                className="simulation-header",
                children=[
                    html.Div("변수", className="sim-header-var"),
                    html.Div("조정", className="sim-header-slider"),
                    html.Div("현재값", className="sim-header-value"),
                    html.Div("", className="sim-header-button")
                ]
            ),
            html.Div(
                className="simulation-body",
                children=[
                    make_simulation_row(feature)
                    for feature in ordered_features
                ]
            ),

            html.Div(
                className="simulation-note",
                children="변수 순서는 모델의 feature_importances_ 기준으로 자동 정렬됩니다."
            )
        ]
    )


# =========================================================
# 6번 결과 게이지 카드
# =========================================================
def make_simulation_result_card(height="360px"):
    return html.Div(
        className="card dashboard-card simulation-result-card",
        style={"minHeight": height},
        children=[
            html.Div(
                className="card-title-row",
                children=[
                    html.Span("6", className="card-number"),
                    html.H3("시뮬레이션 결과 게이지", className="card-title")
                ]
            ),

            dcc.Graph(
                id="simulation-result-gauge",
                figure=make_probability_gauge(0.5),
                config={"displayModeBar": False},
                className="gauge-graph"
            ),

            html.Div(
                id="simulation-result-text",
                className="simulation-result-text"
            )
        ]
    )


# =========================================================
# Layout
# =========================================================

app.layout = html.Div(
    className="page",
    children=[

        # =========================
        # Header
        # =========================
        html.Div(
            className="header",
            children=[
                html.H1("KRWUSD 15분 후 상승/하락 예측 Dashboard",
                    className="main-title"
                ),
                html.Div(
                    "분당 데이터 기반 예측 · 실시간 모니터링 · 시뮬레이션",
                    className="subtitle"
                )
            ]
        ),

        # =========================
        # Row 1
        # =========================
        html.Div(
            className="grid-row row-1",
            style={"gridTemplateColumns": row1_cols},
            children=[
                make_card("현재 예측 게이지", 1, height=row1_height),
                make_card("최근 KRWUSD 흐름", 2, height=row1_height),
                make_model_summary_card(height=row1_height),
            ]
        ),
        # =========================
        # Row 2
        # =========================
        html.Div(
            className="grid-row row-2",
            style={"gridTemplateColumns": row2_cols},
            children=[
                make_card("변수 중요도 Top 5", 4, height=row2_height),
                make_simulation_panel_card(height=row2_height),
                make_simulation_result_card(height=row2_height),
            ]
        ),
    ]
)


@app.callback(
    Output({"type": "sim-value-label", "feature": MATCH}, "children"),
    Input({"type": "sim-slider", "feature": MATCH}, "value"),
    State({"type": "sim-slider", "feature": MATCH}, "id")
)
def update_value_label(value, slider_id):
    feature = slider_id["feature"]

    if value is None:
        value = feature_meta[feature]["latest"]

    std_value = feature_meta[feature]["std"]

    return format_by_std(value, std_value)


# =========================================================
# +/- 기능
# =========================================================
@app.callback(
    Output({"type": "sim-slider", "feature": MATCH}, "value"),
    Input({"type": "sim-minus", "feature": MATCH}, "n_clicks"),
    Input({"type": "sim-plus", "feature": MATCH}, "n_clicks"),
    State({"type": "sim-slider", "feature": MATCH}, "value"),
    State({"type": "sim-slider", "feature": MATCH}, "min"),
    State({"type": "sim-slider", "feature": MATCH}, "max"),
    State({"type": "sim-slider", "feature": MATCH}, "step"),
    prevent_initial_call=True
)
def update_slider_with_buttons(n_minus, n_plus, current_value, min_value, max_value, step):
    if not ctx.triggered_id:
        raise PreventUpdate

    if current_value is None:
        current_value = min_value

    current_value = float(current_value)
    step = float(step)

    if ctx.triggered_id["type"] == "sim-minus":
        new_value = current_value - step
    elif ctx.triggered_id["type"] == "sim-plus":
        new_value = current_value + step
    else:
        return no_update

    new_value = max(float(min_value), min(float(max_value), new_value))

    return new_value
    
# =========================================================
# 시뮬레이션 입력값으로 model.predict_proba 수행
# =========================================================
@app.callback(
    Output("simulation-result-gauge", "figure"),
    Output("simulation-result-text", "children"),
    *[
        Input({"type": "sim-slider", "feature": feature}, "value")
        for feature in ordered_features
    ]
)
def update_simulation_result(*values):
    input_dict = {}

    for feature, value in zip(ordered_features, values):
        if value is None:
            value = feature_meta[feature]["latest"]

        input_dict[feature] = float(value)

    X_sim = pd.DataFrame(
        [[input_dict[col] for col in feature_cols]],
        columns=feature_cols
    )

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_sim)[0]
        class_list = list(model.classes_)

        if 1 in class_list:
            prob_1 = float(proba[class_list.index(1)])
        else:
            prob_1 = float(proba[-1])
    else:
        pred = model.predict(X_sim)[0]
        prob_1 = float(pred)

    fig = make_probability_gauge(prob_1)

    if prob_1 >= 0.5:
        result_label = "상승"
        result_class = "result-up"
    else:
        result_label = "하락"
        result_class = "result-down"

    result_text = html.Div(
        className=result_class,
        children=[
            html.Div(
                f"{result_label} 확률 {prob_1 * 100:.2f}%",
                className="result-main-text"
            ),
            html.Div(
                "판정 기준: Class 1 확률 50% 이상 → 상승",
                className="result-sub-text"
            )
        ]
    )

    return fig, result_text


    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050)