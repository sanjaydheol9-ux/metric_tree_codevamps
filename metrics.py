import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional
from scipy import stats

WEIGHTS = {
    "picking": 0.4,
    "packing": 0.3,
    "dispatch": 0.3,
}

BENCHMARKS = {
    "pick_time":      {"excellent": 10, "good": 20, "poor": 35},
    "pack_time":      {"excellent": 8,  "good": 18, "poor": 30},
    "dispatch_delay": {"excellent": 5,  "good": 15, "poor": 25},
    "error_rate":     {"excellent": 0.1,"good": 0.5,"poor": 1.5},
}

ALERT_THRESHOLDS = {
    "delivery_score": {"warning": 60, "critical": 40},
    "error_rate":     {"warning": 0.8, "critical": 1.5},
    "picking_score":  {"warning": 55, "critical": 35},
    "packing_score":  {"warning": 55, "critical": 35},
    "dispatch_score": {"warning": 55, "critical": 35},
}


@dataclass
class Alert:
    metric: str
    level: str
    value: float
    threshold: float
    message: str


@dataclass
class BenchmarkRating:
    metric: str
    value: float
    rating: str
    percentile_estimate: float


@dataclass
class Forecast:
    metric: str
    next_week_value: float
    trend: str
    slope: float
    r_squared: float
    confidence: str


@dataclass
class WeekMetrics:
    week: int
    picking_score: float
    packing_score: float
    dispatch_score: float
    delivery_score: float
    error_rate: float
    sample_size: int
    alerts: list[Alert] = field(default_factory=list)
    benchmarks: list[BenchmarkRating] = field(default_factory=list)
    forecasts: list[Forecast] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_metric_tree(self) -> dict:
        return {
            "name": "Delivery Timeliness",
            "value": self.delivery_score,
            "status": self._score_status(self.delivery_score),
            "alerts": [a.level for a in self.alerts if a.metric == "delivery_score"],
            "children": [
                {"name": "Picking Efficiency",   "value": self.picking_score,  "weight": WEIGHTS["picking"],  "status": self._score_status(self.picking_score)},
                {"name": "Packing Efficiency",   "value": self.packing_score,  "weight": WEIGHTS["packing"],  "status": self._score_status(self.packing_score)},
                {"name": "Dispatch Performance", "value": self.dispatch_score, "weight": WEIGHTS["dispatch"], "status": self._score_status(self.dispatch_score)},
            ],
        }

    @staticmethod
    def _score_status(score: float) -> str:
        if score >= 75: return "excellent"
        if score >= 60: return "good"
        if score >= 40: return "average"
        return "poor"


def _sigmoid_score(value: float, target: float, scale: float = 10.0) -> float:
    delta = target - value
    return round(100 / (1 + np.exp(-delta / scale)), 2)


def _compute_scores(avg_pick: float, avg_pack: float, avg_dispatch: float) -> tuple[float, float, float, float]:
    picking_score  = _sigmoid_score(avg_pick,     BENCHMARKS["pick_time"]["good"])
    packing_score  = _sigmoid_score(avg_pack,     BENCHMARKS["pack_time"]["good"])
    dispatch_score = _sigmoid_score(avg_dispatch, BENCHMARKS["dispatch_delay"]["good"])
    delivery_score = (
        WEIGHTS["picking"]  * picking_score +
        WEIGHTS["packing"]  * packing_score +
        WEIGHTS["dispatch"] * dispatch_score
    )
    return picking_score, packing_score, dispatch_score, round(delivery_score, 2)


def _generate_alerts(metrics: dict) -> list[Alert]:
    alerts = []
    for metric, thresholds in ALERT_THRESHOLDS.items():
        value = metrics.get(metric)
        if value is None:
            continue

        is_low_good = metric != "error_rate"
        critical, warning = thresholds["critical"], thresholds["warning"]

        if is_low_good:
            level = "critical" if value <= critical else "warning" if value <= warning else None
        else:
            level = "critical" if value >= critical else "warning" if value >= warning else None

        if level:
            threshold = critical if level == "critical" else warning
            alerts.append(Alert(
                metric=metric,
                level=level,
                value=round(value, 2),
                threshold=threshold,
                message=(
                    f"{metric.replace('_', ' ').title()} is {level.upper()}: {value:.2f} "
                    f"({'below' if is_low_good else 'above'} threshold of {threshold})"
                ),
            ))
    return alerts


def _benchmark_metric(name: str, value: float) -> BenchmarkRating:
    thresholds = BENCHMARKS[name]
    lower_is_better = name in ("pick_time", "pack_time", "dispatch_delay", "error_rate")
    v = value

    if lower_is_better:
        if v <= thresholds["excellent"]:   rating, pct = "excellent", 90.0
        elif v <= thresholds["good"]:      rating, pct = "good", 70.0
        elif v <= thresholds["poor"]:      rating, pct = "average", 40.0
        else:                              rating, pct = "poor", 15.0
    else:
        if v >= thresholds["excellent"]:   rating, pct = "excellent", 90.0
        elif v >= thresholds["good"]:      rating, pct = "good", 70.0
        elif v >= thresholds["poor"]:      rating, pct = "average", 40.0
        else:                              rating, pct = "poor", 15.0

    return BenchmarkRating(metric=name, value=round(value, 2), rating=rating, percentile_estimate=pct)


def _forecast_metric(series: pd.Series, metric_name: str) -> Optional[Forecast]:
    if len(series) < 3:
        return None

    x = np.arange(len(series))
    slope, intercept, r_value, _, _ = stats.linregress(x, series.values)
    next_value = intercept + slope * len(series)
    r_squared  = r_value ** 2
    trend      = "stable" if abs(slope) <= 0.5 else ("improving" if slope < 0 else "degrading")
    confidence = "high" if r_squared > 0.7 else "medium" if r_squared > 0.4 else "low"

    return Forecast(
        metric=metric_name,
        next_week_value=round(next_value, 2),
        trend=trend,
        slope=round(slope, 4),
        r_squared=round(r_squared, 4),
        confidence=confidence,
    )


def load_data(path: str = "sample_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"week", "pick_time", "pack_time", "dispatch_delay", "error_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df


def calculate_week_metrics(week: int, df: Optional[pd.DataFrame] = None) -> WeekMetrics:
    if df is None:
        df = load_data()

    df_week = df[df["week"] == week]
    if df_week.empty:
        raise ValueError(f"No data found for week {week}.")

    avg_pick     = df_week["pick_time"].mean()
    avg_pack     = df_week["pack_time"].mean()
    avg_dispatch = df_week["dispatch_delay"].mean()
    error_rate   = df_week["error_count"].sum() / len(df_week)

    picking_score, packing_score, dispatch_score, delivery_score = _compute_scores(avg_pick, avg_pack, avg_dispatch)

    alerts = _generate_alerts({
        "delivery_score": delivery_score,
        "picking_score":  picking_score,
        "packing_score":  packing_score,
        "dispatch_score": dispatch_score,
        "error_rate":     error_rate,
    })

    benchmarks = [
        _benchmark_metric("pick_time",      avg_pick),
        _benchmark_metric("pack_time",      avg_pack),
        _benchmark_metric("dispatch_delay", avg_dispatch),
        _benchmark_metric("error_rate",     error_rate),
    ]

    history   = df[df["week"] <= week].groupby("week")
    forecasts = [
        f for col in ("pick_time", "pack_time", "dispatch_delay")
        if (f := _forecast_metric(history[col].mean().sort_index(), col)) is not None
    ]

    return WeekMetrics(
        week=week,
        picking_score=picking_score,
        packing_score=packing_score,
        dispatch_score=dispatch_score,
        delivery_score=delivery_score,
        error_rate=round(error_rate, 2),
        sample_size=len(df_week),
        alerts=alerts,
        benchmarks=benchmarks,
        forecasts=forecasts,
    )


def compare_weeks(weeks: list[int], df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df is None:
        df = load_data()

    rows = []
    for week in weeks:
        try:
            m = calculate_week_metrics(week, df=df)
            rows.append({
                "week":            m.week,
                "delivery_score":  m.delivery_score,
                "picking_score":   m.picking_score,
                "packing_score":   m.packing_score,
                "dispatch_score":  m.dispatch_score,
                "error_rate":      m.error_rate,
                "sample_size":     m.sample_size,
                "critical_alerts": sum(1 for a in m.alerts if a.level == "critical"),
                "warning_alerts":  sum(1 for a in m.alerts if a.level == "warning"),
            })
        except ValueError:
            continue

    result = pd.DataFrame(rows).set_index("week")
    result["delivery_score_delta"] = result["delivery_score"].diff().round(2)
    return result


def metric_tree(week: int, df: Optional[pd.DataFrame] = None) -> dict:
    return calculate_week_metrics(week, df=df).to_metric_tree()


if __name__ == "__main__":
    df = load_data()
    weeks = sorted(df["week"].unique())
    latest = weeks[-1]
    m = calculate_week_metrics(latest, df=df)

    print(f"\n{'='*55}")
    print(f"  Supply Chain Intelligence Report  —  Week {latest}")
    print(f"{'='*55}\n")
    print(f"  Delivery Score  : {m.delivery_score:>6.2f}")
    print(f"  Picking Score   : {m.picking_score:>6.2f}")
    print(f"  Packing Score   : {m.packing_score:>6.2f}")
    print(f"  Dispatch Score  : {m.dispatch_score:>6.2f}")
    print(f"  Error Rate      : {m.error_rate:>6.2f}")
    print(f"  Sample Size     : {m.sample_size:>6}")

    if m.alerts:
        print(f"\n  Alerts ({len(m.alerts)})")
        for a in m.alerts:
            print(f"     [{a.level.upper()}]  {a.message}")

    if m.forecasts:
        print(f"\n  Forecasts (next week)")
        for f in m.forecasts:
            arrow = "down" if f.trend == "improving" else "up" if f.trend == "degrading" else "-"
            print(f"     {arrow}  {f.metric}: {f.next_week_value:.2f}  [{f.trend}, {f.confidence} confidence, R2={f.r_squared}]")

    print(f"\n  Week-over-Week Comparison")
    print(compare_weeks(weeks, df=df).to_string())

    print(f"\n  Metric Tree")
    print(json.dumps(metric_tree(latest, df=df), indent=4))