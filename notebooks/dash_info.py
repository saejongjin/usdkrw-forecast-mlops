from dash import Dash, html, dcc, Input, Output, State, MATCH
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import math
import os
import sys
import subprocess
import time
from dash import ctx, no_update
from dash.exceptions import PreventUpdate

# =========================================================
# 데이터 및 모델 불러오기 / 갱신 유틸
# =========================================================
DEFAULT_DATA_PATH = "/app/data/processed/market_data_1m_24h_interpolated.csv"
DEFAULT_MODEL_PATH = "/app/data/processed/predict_model.pkl"
DEFAULT_PREPROCESS_PATH = "/app/src/preprocess.py"

DATA_PATH = os.environ.get("DATA_PATH", DEFAULT_DATA_PATH)
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
PREPROCESS_PATH = os.environ.get("PREPROCESS_PATH", DEFAULT_PREPROCESS_PATH)
PREPROCESS_TIMEOUT = int(os.environ.get("PREPROCESS_TIMEOUT", "600"))

# Update 버튼 클릭 시 이 전역 상태를 다시 로딩합니다.
df = pd.DataFrame()
model = None
feature_cols = []
importance_df = pd.DataFrame()
ordered_features = []
importance_lookup = {}
feature_meta = {}
ksj1 = "-"
ksj2 = "-"


def resolve_existing_path(path_value, fallback_candidates):
    """환경변수/기본 경로가 없을 때 로컬 테스트 경로까지 순차 탐색."""
    if path_value and os.path.exists(path_value):
        return path_value

    for candidate in fallback_candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    return path_value


def resolve_data_and_model_paths():
    data_path = resolve_existing_path(
        os.environ.get("DATA_PATH", DEFAULT_DATA_PATH),
        [
            "market_data_1m_24h_interpolated.csv",
            "/mnt/data/market_data_1m_24h_interpolated.csv",
        ],
    )
    model_path = resolve_existing_path(
        os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH),
        [
            "predict_model.pkl",
            "/mnt/data/predict_model.pkl",
        ],
    )
    return data_path, model_path


def resolve_preprocess_path():
    #script_dir = os.path.dirname(os.path.abspath(__file__))

    return resolve_existing_path(
        os.environ.get("PREPROCESS_PATH", DEFAULT_PREPROCESS_PATH),
        [
            DEFAULT_PREPROCESS_PATH,
            #os.path.join(script_dir, "preprocess.py"),
            os.path.join(os.getcwd(), "preprocess.py"),
            "/mnt/data/preprocess.py",
        ],
    )


# =========================================================
# 숫자 표시 유틸
# =========================================================
def decimals_from_std(std_value, sig=2):
    """표준편차 기준으로 표시할 소수점 자리수 결정"""
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


def format_importance(feature):
    value = importance_lookup.get(feature, np.nan)
    if pd.isna(value):
        return "-"
    return f"{float(value):.3f}"


def value_to_percent(value, min_value, max_value):
    """range 값의 위치를 0~100% 문자열로 변환"""
    min_value = float(min_value)
    max_value = float(max_value)
    value = float(value)

    if max_value == min_value:
        return "0%"

    pct = (value - min_value) / (max_value - min_value) * 100
    pct = max(0, min(100, pct))
    return f"{pct:.6f}%"


def refresh_dashboard_state(rebuild_pdp=False):
    """CSV/model을 다시 읽고 dashboard가 사용하는 전역 상태를 재계산합니다."""
    global DATA_PATH, MODEL_PATH
    global df, model, feature_cols, importance_df, ordered_features, importance_lookup, feature_meta
    global ksj1, ksj2, pdp_cache

    DATA_PATH, MODEL_PATH = resolve_data_and_model_paths()

    if not DATA_PATH or not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"DATA_PATH를 찾을 수 없습니다: {DATA_PATH}")
    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH를 찾을 수 없습니다: {MODEL_PATH}")

    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    else:
        feature_cols = [c for c in df.columns if c != "timestamp"]

    feature_cols = [c for c in feature_cols if c in df.columns]

    if hasattr(model, "feature_importances_"):
        raw_importances = list(model.feature_importances_)
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": raw_importances[:len(feature_cols)],
        }).sort_values("importance", ascending=False)
    else:
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": np.nan,
        })

    ordered_features = importance_df["feature"].tolist()
    importance_lookup = dict(zip(importance_df["feature"], importance_df["importance"]))

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
        step = 10 ** (-decimals)

        feature_meta[col] = {
            "std": std_value,
            "decimals": decimals,
            "min": min_val,
            "max": max_val,
            "mid": mid_val,
            "latest": latest_val,
            "step": step,
            "importance": importance_lookup.get(col, np.nan),
        }

    if "timestamp" in df.columns and not df["timestamp"].dropna().empty:
        ksj1 = df["timestamp"].min().strftime("%Y-%m-%d %H:%M")
        ksj2 = df["timestamp"].max().strftime("%Y-%m-%d %H:%M")
    else:
        ksj1 = "-"
        ksj2 = "-"

    if rebuild_pdp and "build_pdp_cache" in globals():
        pdp_cache = build_pdp_cache()

    return {
        "data_path": DATA_PATH,
        "model_path": MODEL_PATH,
        "rows": len(df),
        "latest_timestamp": ksj2,
    }


def run_preprocess_script():
    """Update 버튼에서 preprocess.py를 별도 Python 프로세스로 실행."""
    preprocess_path = resolve_preprocess_path()

    if not preprocess_path or not os.path.exists(preprocess_path):
        return False, f"preprocess.py를 찾을 수 없습니다: {preprocess_path}"

    started = time.time()
    try:
        completed = subprocess.run(
            [sys.executable, preprocess_path],
            cwd=os.path.dirname(preprocess_path) or None,
            capture_output=True,
            text=True,
            timeout=PREPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return False, f"preprocess.py 실행 시간 초과({PREPROCESS_TIMEOUT}s)"
    except Exception as exc:
        return False, f"preprocess.py 실행 실패: {exc}"

    elapsed = time.time() - started
    stdout_tail = (completed.stdout or "").strip().splitlines()[-2:]
    stderr_tail = (completed.stderr or "").strip().splitlines()[-3:]

    if completed.returncode != 0:
        detail = " | ".join(stderr_tail or stdout_tail or [f"returncode={completed.returncode}"])
        return False, f"preprocess.py 오류({elapsed:.1f}s): {detail}"

    detail = " | ".join(stdout_tail) if stdout_tail else "정상 종료"
    return True, f"preprocess.py 완료({elapsed:.1f}s): {detail}"


# 최초 로딩
refresh_dashboard_state(rebuild_pdp=False)


# =========================================================
# Dash App
# =========================================================
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.BOOTSTRAP,
    ],
)


# =========================
# 레이아웃 옵션
# =========================
row1_cols = "1.05fr 1.95fr 1.05fr"
# 5번 패널이 잘리지 않도록 2행 가운데 카드에 조금 더 많은 폭 배정
row2_cols = "1.0fr 2.05fr 1.0fr"

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
                    html.H3(title, className="card-title"),
                ],
            ),
            html.Div(className="card-body-empty"),
        ],
    )


# =========================================================
# 1번 현재 예측 게이지 카드
# - Gauge graph.py의 색상/구간/표시 로직을 대시보드 카드 크기에 맞춰 축소 적용
# =========================================================
def get_latest_up_probability_percent():
    """마지막 행 기준 class 1, 즉 상승 확률(%) 계산."""
    latest = df.tail(1)

    if hasattr(model, "feature_names_in_"):
        model_features = [col for col in list(model.feature_names_in_) if col in df.columns]
    else:
        preferred_features = ["BTC", "JPY", "WTI", "GOLD", "DXY"]
        model_features = [col for col in preferred_features if col in df.columns]

    if not model_features:
        return 50.0

    X_latest = latest[model_features]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_latest)[0]
        class_list = list(model.classes_)
        class_1_index = class_list.index(1) if 1 in class_list else -1
        return float(proba[class_1_index] * 100)

    return float(model.predict(X_latest)[0] * 100)


def make_current_prediction_gauge_figure():
    up_prob = get_latest_up_probability_percent()
    down_prob = 100 - up_prob

    if up_prob >= 50:
        signal = "상승"
        main_prob = up_prob
        sub_prob = down_prob
        signal_color = "#DC2626"
        opposite_text = f"하락확률 {sub_prob:.0f}%"
        opposite_color = "#2563EB"
    else:
        signal = "하락"
        main_prob = down_prob
        sub_prob = up_prob
        signal_color = "#2563EB"
        opposite_text = f"상승확률 {sub_prob:.0f}%"
        opposite_color = "#DC2626"

    fig = go.Figure(
        go.Indicator(
            mode="gauge",
            value=up_prob,
            domain={"x": [0.03, 0.97], "y": [0.22, 0.98]},
            gauge={
                "shape": "angular",
                "axis": {
                    "range": [0, 100],
                    "showticklabels": False,
                    "ticks": "",
                    "tickwidth": 0,
                },
                "bar": {"color": "#e6c3eb", "thickness": 0.30},
                "steps": [
                    {"range": [0, 16.67], "color": "#2563EB"},
                    {"range": [16.67, 33.33], "color": "#5BA9FC"},
                    {"range": [33.33, 50], "color": "#93C5FD"},
                    {"range": [50, 66.67], "color": "#FECACA"},
                    {"range": [66.67, 83.33], "color": "#F87171"},
                    {"range": [83.33, 100], "color": "#B91C1C"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "value": 50,
                },
            },
        )
    )

    fig.add_annotation(
        x=0.5,
        y=0.86,
        text="50%",
        showarrow=False,
        font={"family": "Arial, Noto Sans KR, sans-serif", "size": 12, "color": "#000000"},
    )
    fig.add_annotation(
        x=0.5,
        y=0.52,
        text=f"<b>{signal}확률</b>",
        showarrow=False,
        font={"family": "Arial, Noto Sans KR, sans-serif", "size": 18, "color": signal_color},
    )
    fig.add_annotation(
        x=0.5,
        y=0.34,
        text=f"<b>{main_prob:.0f}%</b>",
        showarrow=False,
        font={"family": "Arial, Noto Sans KR, sans-serif", "size": 44, "color": signal_color},
    )
    fig.add_annotation(
        x=0.5,
        y=0.12,
        text=opposite_text,
        showarrow=False,
        font={"family": "Arial, Noto Sans KR, sans-serif", "size": 13, "color": opposite_color},
    )

    fig.update_layout(
        height=178,
        margin={"l": 4, "r": 4, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, Noto Sans KR, sans-serif"},
    )

    return fig


def make_current_gauge_card(height="260px"):
    return html.Div(
        className="card dashboard-card current-gauge-card",
        style={"minHeight": height},
        children=[
            html.Div(
                className="card-title-row",
                children=[
                    html.Span("1", className="card-number"),
                    html.H3("현재 예측 게이지", className="card-title"),
                ],
            ),
            dcc.Graph(
                id="current-prediction-gauge",
                figure=make_current_prediction_gauge_figure(),
                config={"displayModeBar": False, "responsive": True},
                className="current-gauge-graph",
            ),
        ],
    )


# =========================================================
# 2번 최근 KRWUSD 흐름 카드
# - Line graph.py의 KRW 라인/마지막 값 표시 로직을 카드 크기에 맞춰 적용
# =========================================================
def make_recent_krwusd_line_figure():
    plot_df = df.copy()

    if "timestamp" in plot_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(plot_df["timestamp"]):
            plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"], errors="coerce")
        plot_df = plot_df.dropna(subset=["timestamp"])
        plot_df["time"] = plot_df["timestamp"].dt.strftime("%H:%M")
    else:
        plot_df["time"] = np.arange(len(plot_df)).astype(str)

    if "KRW" not in plot_df.columns or plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text="KRW 데이터 없음",
            showarrow=False,
            font={"size": 14, "color": "#6b7280"},
        )
        fig.update_layout(
            height=178,
            margin={"l": 30, "r": 15, "t": 6, "b": 25},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    y_min = float(plot_df["KRW"].min())
    y_max = float(plot_df["KRW"].max())
    margin = max((y_max - y_min) * 0.2, 0.3)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["time"],
            y=plot_df["KRW"],
            mode="lines",
            name="KRWUSD",
            line={"color": "#0037ff", "width": 3},
            hovertemplate="시간: %{x}<br>KRWUSD: %{y:.3f}<extra></extra>",
        )
    )

    fig.add_annotation(
        x=plot_df["time"].iloc[-1],
        y=plot_df["KRW"].iloc[-1],
        text=f"{plot_df['KRW'].iloc[-1]:.1f}",
        showarrow=False,
        bgcolor="#2563EB",
        bordercolor="#2563EB",
        borderpad=3,
        font={"color": "white", "size": 11},
        xanchor="left",
        xshift=6,
    )

    tick_step = max(len(plot_df) // 6, 1)
    tick_vals = plot_df["time"].iloc[::tick_step]

    fig.update_layout(
        title="",
        template="plotly_white",
        height=178,
        hovermode="closest",
        showlegend=False,
        margin={"l": 44, "r": 40, "t": 6, "b": 34},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, Noto Sans KR, sans-serif", "size": 11},
    )
    fig.update_xaxes(
        tickvals=tick_vals,
        showline=True,
        linecolor="#E5E7EB",
        linewidth=1,
        showgrid=True,
        gridcolor="#E5E7EB",
        tickfont={"size": 10},
    )
    fig.update_yaxes(
        range=[y_min - margin, y_max + margin],
        showline=True,
        linecolor="#E5E7EB",
        linewidth=1.3,
        showgrid=True,
        gridcolor="#E5E7EB",
        tickfont={"size": 10},
    )

    return fig


def make_recent_line_card(height="260px"):
    return html.Div(
        className="card dashboard-card recent-line-card",
        style={"minHeight": height},
        children=[
            html.Div(
                className="card-title-row",
                children=[
                    html.Span("2", className="card-number"),
                    html.H3("최근 KRWUSD 흐름", className="card-title"),
                ],
            ),
            dcc.Graph(
                id="recent-krwusd-line-graph",
                figure=make_recent_krwusd_line_figure(),
                config={"displayModeBar": False, "responsive": True},
                className="recent-line-graph",
            ),
        ],
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
                children=html.I(className=f"bi {icon_class} summary-icon"),
            ),
            html.Div(
                className="summary-text-box",
                children=[
                    html.Div(label, className="summary-label-modern"),
                    html.Div(value, className="summary-value-modern"),
                ],
            ),
        ],
    )


def make_model_summary_content():
    if "timestamp" in df.columns and not df["timestamp"].dropna().empty:
        now_time = df["timestamp"].max()
        pred_time = now_time + timedelta(minutes=15)
        now_str = now_time.strftime("%Y-%m-%d %H:%M:%S")
        pred_str = pred_time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        now_str = "-"
        pred_str = "-"

    return [
        summary_item("bi-cpu", "모델명", "RandomForest Classifier"),
        summary_item("bi-clock", "예측 시점", f"{pred_str} | 15분 후"),
        summary_item("bi-database", "최신 데이터", now_str),
        summary_item("bi-calendar3", "학습 데이터 범위", f"{ksj1} ~ {ksj2}"),
    ]


# =========================
# 모델 요약 카드
# =========================
def make_model_summary_card(height="260px"):
    return html.Div(
        className="card dashboard-card model-summary-card",
        style={"minHeight": height},
        children=[
            html.Div(
                className="card-title-row",
                children=[
                    html.Span("3", className="card-number"),
                    html.H3("모델 요약", className="card-title"),
                ],
            ),
            html.Div(
                id="model-summary-content",
                className="model-summary-modern",
                children=make_model_summary_content(),
            ),
        ],
    )


# =========================================================
# Gauge Chart 생성 함수
# - 6번 시뮬레이션 결과 게이지도 1번 게이지 디자인과 동일한 계열 사용
# =========================================================
def make_probability_gauge(prob_1):
    """
    prob_1: Class 1 확률.
        - 0~1 범위 입력을 기본으로 사용
        - 혹시 0~100 값이 들어와도 안전하게 처리
    """
    up_prob = float(prob_1)
    if up_prob <= 1.0:
        up_prob *= 100

    up_prob = max(0.0, min(100.0, up_prob))
    down_prob = 100.0 - up_prob

    if up_prob >= 50:
        signal = "상승"
        main_prob = up_prob
        sub_prob = down_prob
        signal_color = "#DC2626"
        opposite_text = f"하락확률 {sub_prob:.0f}%"
        opposite_color = "#2563EB"
    else:
        signal = "하락"
        main_prob = down_prob
        sub_prob = up_prob
        signal_color = "#2563EB"
        opposite_text = f"상승확률 {sub_prob:.0f}%"
        opposite_color = "#DC2626"

    fig = go.Figure(
        go.Indicator(
            mode="gauge",
            value=up_prob,
            domain={"x": [0.03, 0.97], "y": [0.22, 0.98]},
            gauge={
                "shape": "angular",
                "axis": {
                    "range": [0, 100],
                    "showticklabels": False,
                    "ticks": "",
                    "tickwidth": 0,
                },
                # 1번 Gauge graph.py 스타일의 현재값 표시 바
                "bar": {"color": "#e6c3eb", "thickness": 0.30},
                # 1번 Gauge graph.py의 6단계 구간 색상
                "steps": [
                    {"range": [0, 16.67], "color": "#2563EB"},
                    {"range": [16.67, 33.33], "color": "#5BA9FC"},
                    {"range": [33.33, 50], "color": "#93C5FD"},
                    {"range": [50, 66.67], "color": "#FECACA"},
                    {"range": [66.67, 83.33], "color": "#F87171"},
                    {"range": [83.33, 100], "color": "#B91C1C"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "value": 50,
                },
            },
        )
    )

    fig.add_annotation(
        x=0.5,
        y=0.86,
        text="50%",
        showarrow=False,
        font={"family": "Arial, Noto Sans KR, sans-serif", "size": 14, "color": "#000000"},
    )
    fig.add_annotation(
        x=0.5,
        y=0.52,
        text=f"<b>{signal}확률</b>",
        showarrow=False,
        font={"family": "Arial, Noto Sans KR, sans-serif", "size": 22, "color": signal_color},
    )
    fig.add_annotation(
        x=0.5,
        y=0.34,
        text=f"<b>{main_prob:.0f}%</b>",
        showarrow=False,
        font={"family": "Arial, Noto Sans KR, sans-serif", "size": 54, "color": signal_color},
    )
    fig.add_annotation(
        x=0.5,
        y=0.12,
        text=opposite_text,
        showarrow=False,
        font={"family": "Arial, Noto Sans KR, sans-serif", "size": 16, "color": opposite_color},
    )

    fig.update_layout(
        height=292,
        margin={"l": 4, "r": 4, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, Noto Sans KR, sans-serif"},
    )

    return fig


# =========================================================
# PDP 계산 및 4번 카드
# =========================================================
PDP_GRID_SIZE = 36
PDP_SAMPLE_SIZE = 600


def _get_class_1_proba(X_input):
    """모델에서 class 1 확률을 안전하게 추출."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input)
        class_list = list(model.classes_)
        class_1_index = class_list.index(1) if 1 in class_list else -1
        return proba[:, class_1_index]

    pred = model.predict(X_input)
    return np.asarray(pred, dtype=float)


def build_pdp_cache():
    """각 변수별 PDP 값을 미리 계산."""
    X_base = df[feature_cols].dropna().copy()

    if len(X_base) > PDP_SAMPLE_SIZE:
        X_base = X_base.sample(n=PDP_SAMPLE_SIZE, random_state=42)

    X_base = X_base.reset_index(drop=True)
    n_base = len(X_base)
    cache = {}

    for feature in ordered_features:
        meta = feature_meta[feature]
        min_val = float(meta["min"])
        max_val = float(meta["max"])

        if max_val == min_val:
            grid = np.array([min_val])
        else:
            grid = np.linspace(min_val, max_val, PDP_GRID_SIZE)

        # grid별 데이터를 한 번에 만들어 predict_proba를 한 번만 호출
        X_rep = pd.concat([X_base] * len(grid), ignore_index=True)
        X_rep[feature] = np.repeat(grid, n_base)

        probs = _get_class_1_proba(X_rep)
        pdp_values = probs.reshape(len(grid), n_base).mean(axis=1)

        cache[feature] = {
            "x": grid,
            "y": pdp_values,
        }

    return cache


pdp_cache = build_pdp_cache()


def make_pdp_figure(feature):
    """선택 변수의 PDP Plotly Figure 생성."""
    if feature not in pdp_cache:
        feature = ordered_features[0]

    x_values = pdp_cache[feature]["x"]
    y_values = pdp_cache[feature]["y"] * 100
    std_value = feature_meta[feature]["std"]

    hover_x = [format_by_std(x, std_value) for x in x_values]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers",
            showlegend=False,
            line={"width": 3},
            marker={"size": 5},
            hovertemplate=(
                f"<b>{feature}</b>: %{{customdata}}<br>"
                "평균 상승 확률: %{y:.2f}%"
                "<extra></extra>"
            ),
            customdata=hover_x,
        )
    )

    fig.update_layout(
        height=285,
        margin=dict(l=42, r=14, t=16, b=42),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, Noto Sans KR, sans-serif", "size": 15},
        hovermode="closest",
        showlegend=False,
    )
    fig.update_xaxes(
        title_text=feature,
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False,
        tickfont={"size": 15},
        title_font={"size": 16},
    )
    fig.update_yaxes(
        title_text="상승 확률(%)",
        range=[0, 100],
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False,
        tickfont={"size": 15},
        title_font={"size": 16},
    )

    return fig


def make_pdp_card(height="360px"):
    return html.Div(
        className="card dashboard-card pdp-card",
        style={"minHeight": height},
        children=[
            html.Div(
                className="pdp-title-row",
                children=[
                    html.Div(
                        className="pdp-title-left",
                        children=[
                            html.Span("4", className="card-number"),
                            html.H3("각 변수별 PDP 그래프", className="card-title"),
                        ],
                    ),
                    dcc.Dropdown(
                        id="pdp-feature-dropdown",
                        options=[
                            {"label": feature, "value": feature}
                            for feature in ordered_features
                        ],
                        value=ordered_features[0],
                        clearable=False,
                        searchable=False,
                        className="pdp-dropdown",
                    ),
                ],
            ),
            dcc.Graph(
                id="pdp-graph",
                figure=make_pdp_figure(ordered_features[0]),
                config={"displayModeBar": False, "responsive": True},
                className="pdp-graph",
            ),
        ],
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

    initial_pct = value_to_percent(latest_val, min_val, max_val)

    return html.Div(
        className="sim-row",
        children=[
            # 1열: 변수명
            html.Div(feature, className="sim-var-name"),

            # 2열: 변수 중요도
            html.Div(format_importance(feature), className="sim-importance-value"),

            # 3열: 커스텀 조정바
            html.Div(
                className="sim-range-wrap",
                children=[
                    html.Div(
                        className="sim-range-track",
                        children=[
                            html.Div(
                                id={"type": "sim-range-fill", "feature": feature},
                                className="sim-range-fill",
                                style={"width": initial_pct},
                            )
                        ],
                    ),
                    html.Div(
                        id={"type": "sim-range-thumb", "feature": feature},
                        className="sim-range-thumb-dot",
                        style={"left": initial_pct},
                    ),
                    dcc.Input(
                        id={"type": "sim-range", "feature": feature},
                        type="range",
                        min=min_val,
                        max=max_val,
                        step=step,
                        value=latest_val,
                        className="sim-range-input",
                    ),
                    html.Div(
                        className="sim-range-marks",
                        children=[
                            html.Span(
                                format_by_std(min_val, std_value),
                                className="sim-range-mark sim-range-mark-left",
                            ),
                            html.Span(
                                format_by_std(mid_val, std_value),
                                className="sim-range-mark sim-range-mark-center",
                            ),
                            html.Span(
                                format_by_std(max_val, std_value),
                                className="sim-range-mark sim-range-mark-right",
                            ),
                        ],
                    ),
                ],
            ),

            # 4열: 현재값 표시
            html.Div(
                id={"type": "sim-value-label", "feature": feature},
                className="sim-value-label",
                children=format_by_std(latest_val, std_value),
            ),

            # 5열: +/- 버튼
            html.Div(
                className="sim-step-buttons",
                children=[
                    html.Button(
                        "−",
                        id={"type": "sim-minus", "feature": feature},
                        n_clicks=0,
                        className="sim-step-button",
                    ),
                    html.Button(
                        "+",
                        id={"type": "sim-plus", "feature": feature},
                        n_clicks=0,
                        className="sim-step-button",
                    ),
                ],
            ),
        ],
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
                    html.H3("시뮬레이션 패널", className="card-title"),
                ],
            ),
            html.Div(
                className="simulation-header",
                children=[
                    html.Div("변수", className="sim-header-var"),
                    html.Div("중요도", className="sim-header-importance"),
                    html.Div("조정", className="sim-header-slider"),
                    html.Div("현재값", className="sim-header-value"),
                    html.Div("", className="sim-header-button"),
                ],
            ),
            html.Div(
                id="simulation-body",
                className="simulation-body",
                children=[make_simulation_row(feature) for feature in ordered_features],
            ),
            html.Div(
                className="simulation-note",
                children="변수 순서는 모델의 feature_importances_ 기준으로 자동 정렬됩니다.",
            ),
        ],
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
                    html.H3("시뮬레이션 결과 게이지", className="card-title"),
                ],
            ),
            dcc.Graph(
                id="simulation-result-gauge",
                figure=make_probability_gauge(0.5),
                config={"displayModeBar": False, "responsive": True},
                className="gauge-graph",
            ),
            html.Div(id="simulation-result-text", className="simulation-result-text"),
        ],
    )


# =========================================================
# Layout
# =========================================================
app.layout = html.Div(
    className="page",
    children=[
        dcc.Store(id="dashboard-refresh-store", data={"version": 0}),
        html.Div(
            className="update-control",
            children=[
                html.Button("Update", id="update-button", n_clicks=0, className="update-button"),
                html.Div(id="update-status", className="update-status"),
            ],
        ),
        html.Div(
            className="header",
            children=[
                html.H1("KRWUSD 15분 후 상승/하락 예측 Dashboard", className="main-title"),
                html.Div(
                    "분당 데이터 기반 예측 · 실시간 모니터링 · 시뮬레이션",
                    className="subtitle",
                ),
            ],
        ),
        html.Div(
            className="grid-row row-1",
            style={"gridTemplateColumns": row1_cols},
            children=[
                make_current_gauge_card(height=row1_height),
                make_recent_line_card(height=row1_height),
                make_model_summary_card(height=row1_height),
            ],
        ),
        html.Div(
            className="grid-row row-2",
            style={"gridTemplateColumns": row2_cols},
            children=[
                make_pdp_card(height=row2_height),
                make_simulation_panel_card(height=row2_height),
                make_simulation_result_card(height=row2_height),
            ],
        ),
    ],
)


# =========================================================
# Update 버튼: preprocess.py 실행 후 dashboard 상태 재로딩
# =========================================================
@app.callback(
    Output("current-prediction-gauge", "figure"),
    Output("recent-krwusd-line-graph", "figure"),
    Output("model-summary-content", "children"),
    Output("pdp-feature-dropdown", "options"),
    Output("pdp-feature-dropdown", "value"),
    Output("simulation-body", "children"),
    Output("update-status", "children"),
    Output("update-status", "className"),
    Output("dashboard-refresh-store", "data"),
    Input("update-button", "n_clicks"),
    State("pdp-feature-dropdown", "value"),
    State("dashboard-refresh-store", "data"),
    prevent_initial_call=True,
    running=[(Output("update-button", "disabled"), True, False)],
)
def run_update_and_refresh(n_clicks, selected_pdp_feature, refresh_data):
    if not n_clicks:
        raise PreventUpdate

    ok, preprocess_message = run_preprocess_script()

    # preprocess가 PostgreSQL 등 외부 의존성 때문에 실패하더라도,
    # CSV가 이미 갱신되었을 수 있으므로 데이터 재로딩은 한 번 시도합니다.
    try:
        refresh_dashboard_state(rebuild_pdp=True)
        reload_ok = True
        reload_message = f"데이터 재로딩 완료: {ksj2}"
    except Exception as exc:
        reload_ok = False
        reload_message = f"데이터 재로딩 실패: {exc}"

    if ordered_features:
        dropdown_options = [{"label": feature, "value": feature} for feature in ordered_features]
        dropdown_value = selected_pdp_feature if selected_pdp_feature in ordered_features else ordered_features[0]
    else:
        dropdown_options = []
        dropdown_value = None

    version = 0
    if isinstance(refresh_data, dict):
        version = int(refresh_data.get("version", 0) or 0)

    new_refresh_data = {
        "version": version + 1,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "preprocess_ok": ok,
        "reload_ok": reload_ok,
    }

    if ok and reload_ok:
        status_text = f"Updated · {ksj2}"
        status_class = "update-status update-status-ok"
    elif reload_ok:
        status_text = f"Reloaded with warning · {preprocess_message}"
        status_class = "update-status update-status-warning"
    else:
        status_text = f"Update failed · {preprocess_message} · {reload_message}"
        status_class = "update-status update-status-error"

    return (
        make_current_prediction_gauge_figure(),
        make_recent_krwusd_line_figure(),
        make_model_summary_content(),
        dropdown_options,
        dropdown_value,
        [make_simulation_row(feature) for feature in ordered_features],
        status_text,
        status_class,
        new_refresh_data,
    )


# =========================================================
# 4번 PDP Dropdown callback
# =========================================================
@app.callback(
    Output("pdp-graph", "figure"),
    Input("pdp-feature-dropdown", "value"),
    Input("dashboard-refresh-store", "data"),
)
def update_pdp_graph(selected_feature, _refresh_data):
    if not ordered_features:
        return go.Figure()
    if selected_feature is None or selected_feature not in ordered_features:
        selected_feature = ordered_features[0]
    return make_pdp_figure(selected_feature)


# =========================================================
# 조정바 값 변경 시: 현재값/채움/손잡이 갱신
# =========================================================
@app.callback(
    Output({"type": "sim-value-label", "feature": MATCH}, "children"),
    Output({"type": "sim-range-fill", "feature": MATCH}, "style"),
    Output({"type": "sim-range-thumb", "feature": MATCH}, "style"),
    Input({"type": "sim-range", "feature": MATCH}, "value"),
    State({"type": "sim-range", "feature": MATCH}, "id"),
    State({"type": "sim-range", "feature": MATCH}, "min"),
    State({"type": "sim-range", "feature": MATCH}, "max"),
)
def update_range_visual(value, range_id, min_value, max_value):
    feature = range_id["feature"]

    if value is None:
        value = feature_meta[feature]["latest"]

    value = float(value)
    std_value = feature_meta[feature]["std"]
    pct = value_to_percent(value, min_value, max_value)

    return (
        format_by_std(value, std_value),
        {"width": pct},
        {"left": pct},
    )


# =========================================================
# +/- 기능
# =========================================================
@app.callback(
    Output({"type": "sim-range", "feature": MATCH}, "value"),
    Input({"type": "sim-minus", "feature": MATCH}, "n_clicks"),
    Input({"type": "sim-plus", "feature": MATCH}, "n_clicks"),
    State({"type": "sim-range", "feature": MATCH}, "value"),
    State({"type": "sim-range", "feature": MATCH}, "min"),
    State({"type": "sim-range", "feature": MATCH}, "max"),
    State({"type": "sim-range", "feature": MATCH}, "step"),
    prevent_initial_call=True,
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

    feature = ctx.triggered_id["feature"]
    decimals = feature_meta[feature]["decimals"]
    return round(new_value, decimals)


# =========================================================
# 시뮬레이션 입력값으로 model.predict_proba 수행
# =========================================================
SIM_CALLBACK_FEATURES = ordered_features.copy()


@app.callback(
    Output("simulation-result-gauge", "figure"),
    Output("simulation-result-text", "children"),
    *([
        Input({"type": "sim-range", "feature": feature}, "value")
        for feature in SIM_CALLBACK_FEATURES
    ] + [Input("dashboard-refresh-store", "data")]),
)
def update_simulation_result(*values_and_refresh):
    values = values_and_refresh[:len(SIM_CALLBACK_FEATURES)]
    input_dict = {}

    for feature, value in zip(SIM_CALLBACK_FEATURES, values):
        if feature not in feature_meta:
            continue
        if value is None:
            value = feature_meta[feature]["latest"]
        input_dict[feature] = float(value)

    for feature in feature_cols:
        if feature not in input_dict:
            input_dict[feature] = float(feature_meta.get(feature, {}).get("latest", df[feature].iloc[-1]))

    X_sim = pd.DataFrame(
        [[input_dict[col] for col in feature_cols]],
        columns=feature_cols,
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

    # 1번 게이지 스타일과 동일하게 게이지 내부 annotation에 결과 텍스트가 포함됩니다.
    # 별도의 하단 텍스트는 중복을 피하기 위해 비워둡니다.
    result_text = ""

    return fig, result_text


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8050")),
    )
