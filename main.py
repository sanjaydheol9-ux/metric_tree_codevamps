from contextlib import asynccontextmanager
from typing import Optional, List
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import engine, Base, SessionLocal
from models import WeeklyMetrics

from metrics import calculate_week_metrics, metric_tree, compare_weeks
from root_cause import root_cause_analysis
from simulation import simulate_load
from model import detect_anomalies
from Aiservice import generate_week_insights

Base.metadata.create_all(bind=engine)

logging.basicConfig(level=logging.INFO)

class AppState:
    df: Optional[pd.DataFrame] = None

state = AppState()

def load_from_db():
    db: Session = SessionLocal()
    records = db.query(WeeklyMetrics).all()
    db.close()

    return pd.DataFrame([{
        "week": r.week,
        "delivery_score": r.delivery_score,
        "accuracy_score": r.accuracy_score,
        "dispatch_score": r.dispatch_score,
        "warehouse_score": r.warehouse_score,
        "on_time_score": r.on_time_score,
    } for r in records])

@asynccontextmanager
async def lifespan(app: FastAPI):
    state.df = load_from_db()
    yield
    state.df = None

app = FastAPI(
    title="Supply Chain Intelligence API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulateRequest(BaseModel):
    order_increase_pct: float

class HealthResponse(BaseModel):
    status: str
    weeks_loaded: int
    total_records: int

class AIInsightsResponse(BaseModel):
    status: str
    summary: str
    bottleneck: Optional[str] = None
    root_cause: Optional[str] = None
    recommendations: List[str]

def _get_df():
    if state.df is None:
        raise HTTPException(status_code=503, detail="Data not loaded.")
    return state.df

def _validate_week(week: int, df: pd.DataFrame):
    if week not in df["week"].values:
        raise HTTPException(status_code=404, detail="Week not found.")

@app.get("/health", response_model=HealthResponse)
def health():
    df = _get_df()
    return HealthResponse(
        status="ok",
        weeks_loaded=df["week"].nunique(),
        total_records=len(df),
    )

@app.get("/weeks")
def list_weeks():
    df = _get_df()
    return {"weeks": sorted(df["week"].unique().tolist())}

@app.get("/metrics/{week}")
def get_metrics(week: int):
    df = _get_df()
    _validate_week(week, df)
    return calculate_week_metrics(week, df=df).to_dict()

@app.get("/metrics/{week}/tree")
def get_metric_tree(week: int):
    df = _get_df()
    _validate_week(week, df)
    return metric_tree(week, df=df)

@app.get("/metrics/compare")
def get_comparison(weeks: str = Query(...)):
    df = _get_df()
    week_list = [int(w.strip()) for w in weeks.split(",")]
    for w in week_list:
        _validate_week(w, df)
    result = compare_weeks(week_list, df=df)
    return result.reset_index().to_dict(orient="records")

@app.get("/root-cause")
def get_root_cause(current_week: int, previous_week: int):
    df = _get_df()
    _validate_week(current_week, df)
    _validate_week(previous_week, df)
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

@app.post("/simulate/{week}")
def simulate(week: int, body: SimulateRequest):
    df = _get_df()
    _validate_week(week, df)
    result = simulate_load(week, body.order_increase_pct, df=df)
    return {
        "week": week,
        "order_increase_pct": body.order_increase_pct,
        "baseline_delivery": calculate_week_metrics(week, df=df).delivery_score,
        "simulated_delivery": result.delivery_score,
        "delivery_delta": result.delivery_delta,
        "risk_level": result.risk_level,
    }

@app.get("/anomalies")
def get_anomalies(week: Optional[int] = None):
    df = _get_df()
    if week is not None:
        _validate_week(week, df)
    report = detect_anomalies(df=df, week_filter=week)
    return {
        "total_records": report.total_records,
        "anomaly_count": report.anomaly_count,
        "anomaly_rate": round(report.anomaly_rate, 4),
        "most_common_trigger": report.most_common_trigger,
    }

@app.get("/ai/insights", response_model=AIInsightsResponse)
def ai_insights(current_week: int, previous_week: int):
    df = _get_df()
    _validate_week(current_week, df)
    _validate_week(previous_week, df)
    if current_week == previous_week:
        raise HTTPException(status_code=400, detail="Weeks must be different.")
    result = generate_week_insights(current_week, previous_week)
    return AIInsightsResponse(**result)
