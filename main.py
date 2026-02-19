
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from Aiservice import generate_week_insights


logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Supply Chain Intelligence API",
    version="2.0.0",
)


class AIInsightsResponse(BaseModel):
    status: str
    summary: str
    bottleneck: Optional[str] = None
    root_cause: Optional[str] = None
    recommendations: List[str] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ai/insights", response_model=AIInsightsResponse, tags=["AI"])
def ai_insights(
    current_week: int = Query(..., ge=1),
    previous_week: int = Query(..., ge=1),
):
    if current_week == previous_week:
        raise HTTPException(
            status_code=400,
            detail="current_week and previous_week must be different.",
        )

    try:
        result = generate_week_insights(current_week, previous_week)
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    if result.get("status") == "Error":
        raise HTTPException(status_code=502, detail=result.get("summary"))

    return AIInsightsResponse(**result)
=======
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from metrics import load_data, calculate_week_metrics, metric_tree, compare_weeks
from root_cause import root_cause_analysis
from simulation import simulate_load, stress_test
from model import detect_anomalies, detect_anomalies_by_week


class AppState:
    df: Optional[pd.DataFrame] = None

state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.df = load_data()
    yield
    state.df = None


app = FastAPI(
    title="Supply Chain Intelligence API",
    description="Real-time delivery performance metrics, anomaly detection, root cause analysis, and load simulation.",
    version="1.0.0",
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
    picking_elasticity: float = 1.0
    dispatch_elasticity: float = 1.2


class HealthResponse(BaseModel):
    status: str
    weeks_loaded: int
    total_records: int


def _get_df() -> pd.DataFrame:
    if state.df is None:
        raise HTTPException(status_code=503, detail="Data not loaded.")
    return state.df


def _validate_week(week: int, df: pd.DataFrame) -> None:
    if week not in df["week"].values:
        available = sorted(df["week"].unique().tolist())
        raise HTTPException(
            status_code=404,
            detail=f"Week {week} not found. Available weeks: {available}",
        )


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    df = _get_df()
    return HealthResponse(
        status="ok",
        weeks_loaded=df["week"].nunique(),
        total_records=len(df),
    )


@app.get("/weeks", tags=["System"])
def list_weeks():
    df = _get_df()
    return {"weeks": sorted(df["week"].unique().tolist())}


@app.get("/metrics/{week}", tags=["Metrics"])
def get_metrics(week: int):
    df = _get_df()
    _validate_week(week, df)
    try:
        return calculate_week_metrics(week, df=df).to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/{week}/tree", tags=["Metrics"])
def get_metric_tree(week: int):
    df = _get_df()
    _validate_week(week, df)
    try:
        return metric_tree(week, df=df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/compare", tags=["Metrics"])
def get_comparison(
    weeks: str = Query(..., description="Comma-separated week numbers e.g. 1,2,3")
):
    df = _get_df()
    try:
        week_list = [int(w.strip()) for w in weeks.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="weeks must be comma-separated integers.")

    for w in week_list:
        _validate_week(w, df)

    result = compare_weeks(week_list, df=df)
    return result.reset_index().to_dict(orient="records")


@app.get("/root-cause", tags=["Analysis"])
def get_root_cause(
    current_week: int = Query(...),
    previous_week: int = Query(...),
):
    df = _get_df()
    _validate_week(current_week, df)
    _validate_week(previous_week, df)
    try:
        report = root_cause_analysis(current_week, previous_week, df=df)
        return {
            "kpi":              report.kpi,
            "current_week":     report.current_week,
            "previous_week":    report.previous_week,
            "previous_score":   report.previous_score,
            "current_score":    report.current_score,
            "total_drop":       report.total_drop,
            "main_driver":      report.main_driver,
            "verdict":          report.verdict,
            "drivers": [
                {
                    "metric":          d.metric,
                    "previous":        d.previous,
                    "current":         d.current,
                    "change":          d.change,
                    "weight":          d.weight,
                    "weighted_impact": d.weighted_impact,
                    "direction":       d.direction,
                }
                for d in report.drivers
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate/{week}", tags=["Simulation"])
def simulate(week: int, body: SimulateRequest):
    df = _get_df()
    _validate_week(week, df)
    try:
        result = simulate_load(
            week,
            body.order_increase_pct,
            df=df,
        )
        return {
            "week":                 week,
            "order_increase_pct":   body.order_increase_pct,
            "baseline_delivery":    calculate_week_metrics(week, df=df).delivery_score,
            "simulated_delivery":   result.delivery_score,
            "simulated_picking":    result.picking_score,
            "simulated_dispatch":   result.dispatch_score,
            "delivery_delta":       result.delivery_delta,
            "risk_level":           result.risk_level,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate/{week}/stress", tags=["Simulation"])
def stress(
    week: int,
    step: float = Query(10.0, description="Load increment step (%)"),
    max_increase: float = Query(100.0, description="Max load increase (%)"),
):
    df = _get_df()
    _validate_week(week, df)
    try:
        report = stress_test(week, step=step, max_increase=max_increase, df=df)
        return {
            "base_week":          report.base_week,
            "baseline_delivery":  report.baseline_delivery,
            "breaking_point_pct": report.breaking_point_pct,
            "scenarios": [
                {
                    "label":              s.label,
                    "order_increase_pct": s.order_increase_pct,
                    "delivery_score":     s.delivery_score,
                    "delivery_delta":     s.delivery_delta,
                    "risk_level":         s.risk_level,
                }
                for s in report.scenarios
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anomalies", tags=["Anomalies"])
def get_anomalies(week: Optional[int] = Query(None, description="Filter by week")):
    df = _get_df()
    if week is not None:
        _validate_week(week, df)
    try:
        report = detect_anomalies(df=df, week_filter=week)
        return {
            "total_records":      report.total_records,
            "anomaly_count":      report.anomaly_count,
            "anomaly_rate":       round(report.anomaly_rate, 4),
            "most_common_trigger": report.most_common_trigger,
            "severity_breakdown": report.severity_breakdown,
            "anomalies": [
                {
                    "week":                a.week,
                    "row_index":           a.row_index,
                    "anomaly_score":       a.anomaly_score,
                    "severity":            a.severity,
                    "triggered_features":  a.triggered_features,
                    "values":              a.values,
                    "explanation":         a.explanation,
                }
                for a in report.anomalies
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anomalies/by-week", tags=["Anomalies"])
def get_anomalies_by_week():
    df = _get_df()
    try:
        reports = detect_anomalies_by_week(df=df)
        return {
            week: {
                "anomaly_count":      r.anomaly_count,
                "anomaly_rate":       round(r.anomaly_rate, 4),
                "severity_breakdown": r.severity_breakdown,
                "most_common_trigger": r.most_common_trigger,
            }
            for week, r in reports.items()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)
