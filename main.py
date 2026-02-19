import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from AIservice import generate_week_insights


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
        raise HTTPException(status_code=400, detail="current_week and previous_week must be different.")

    try:
        result = generate_week_insights(current_week, previous_week)
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    if result.get("status") == "Error":
        raise HTTPException(status_code=502, detail=result.get("summary"))

    return AIInsightsResponse(**result)
