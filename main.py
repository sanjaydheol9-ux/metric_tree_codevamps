from contextlib import asynccontextmanager
from typing import Optional, List
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from metrics import calculate_week_metrics, metric_tree, compare_weeks
from root_cause import root_cause_analysis
from simulation import simulate_load
from model import detect_anomalies
from Aiservice import generate_week_insights


# -----------------------
# LOAD CSV
# -----------------------

class AppState:
    df: Optional[pd.DataFrame] = None

state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.df = pd.read_csv("sample_Data.csv")
    yield
    state.df = None


app = FastAPI(
    title="Supply Chain Intelligence API",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------
# MODELS
# -----------------------

class SimulateRequest(BaseModel):
    order_increase_pct: float


class AIInsightsResponse(BaseModel):
    status: str
    summary: str
    bottleneck: Optional[str] = None
    root_cause: Optional[str] = None
    recommendations: List[str]


# -----------------------
# HELPERS
# -----------------------

def get_df():
    if state.df is None:
        raise HTTPException(status_code=503, detail="Data not loaded.")
    return state.df


def validate_week(week: int, df: pd.DataFrame):
    if week not in df["week"].unique():
        raise HTTPException(status_code=404, detail="Week not found.")


# -----------------------
# SYSTEM
# -----------------------

@app.get("/health")
def health():
    df = get_df()
    return {
        "status": "ok",
        "weeks_loaded": df["week"].nunique(),
        "total_records": len(df)
    }


@app.get("/weeks")
def weeks():
    df = get_df()
    return {"weeks": sorted(df["week"].unique().tolist())}


# -----------------------
# METRICS
# -----------------------

@app.get("/metrics/{week}")
def metrics(week: int):
    df = get_df()
    validate_week(week, df)
    return calculate_week_metrics(week, df=df).to_dict()


@app.get("/metrics/{week}/tree")
def tree(week: int):
    df = get_df()
    validate_week(week, df)
    return metric_tree(week, df=df)


@app.get("/metrics/compare")
def compare(weeks: str = Query(...)):
    df = get_df()
    week_list = [int(w.strip()) for w in weeks.split(",")]

    for w in week_list:
        validate_week(w, df)

    result = compare_weeks(week_list, df=df)
    return result.reset_index().to_dict(orient="records")


# -----------------------
# ROOT CAUSE
# -----------------------

@app.get("/root-cause")
def root_cause(current_week: int, previous_week: int):
    df = get_df()
    validate_week(current_week, df)
    validate_week(previous_week, df)

    report = root_cause_analysis(current_week, previous_week, df=df)

    return {
        "kpi": report.kpi,
        "current_week": report.current_week,
        "previous_week": report.previous_week,
        "previous_score": report.previous_score,
        "current_score": report.current_score,
        "total_drop": report.total_drop,
        "main_driver": report.main_driver,
        "verdict": report.verdict,
    }


# -----------------------
# SIMULATION
# -----------------------

@app.post("/simulate/{week}")
def simulate(week: int, body: SimulateRequest):
    df = get_df()
    validate_week(week, df)

    result = simulate_load(week, body.order_increase_pct, df=df)

    return {
        "week": week,
        "order_increase_pct": body.order_increase_pct,
        "baseline_delivery": calculate_week_metrics(week, df=df).delivery_score,
        "simulated_delivery": result.delivery_score,
        "delivery_delta": result.delivery_delta,
        "risk_level": result.risk_level,
    }


# -----------------------
# ANOMALIES
# -----------------------

@app.get("/anomalies")
def anomalies(week: Optional[int] = None):
    df = get_df()

    if week is not None:
        validate_week(week, df)

    report = detect_anomalies(df=df, week_filter=week)

    return {
        "total_records": report.total_records,
        "anomaly_count": report.anomaly_count,
        "anomaly_rate": round(report.anomaly_rate, 4),
        "most_common_trigger": report.most_common_trigger,
    }


# -----------------------
# AI INSIGHTS
# -----------------------

@app.get("/ai/insights", response_model=AIInsightsResponse)
def ai_insights(current_week: int, previous_week: int):
    df = get_df()

    validate_week(current_week, df)
    validate_week(previous_week, df)

    if current_week == previous_week:
        raise HTTPException(status_code=400, detail="Weeks must be different.")

    result = generate_week_insights(current_week, previous_week)

    return AIInsightsResponse(**result)
