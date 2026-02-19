from dataclasses import dataclass
from typing import Optional
import pandas as pd

from metrics import calculate_week_metrics, WEIGHTS


@dataclass
class ScenarioResult:
    label: str
    order_increase_pct: float
    picking_score: float
    packing_score: float
    dispatch_score: float
    delivery_score: float
    delivery_delta: float
    risk_level: str


@dataclass
class LoadSimulationReport:
    kpi: str
    base_week: int
    baseline_delivery: float
    scenarios: list[ScenarioResult]
    breaking_point_pct: Optional[float]

    def summary(self) -> str:
        lines = [
            f"Load Simulation Report: {self.kpi}",
            f"  Base Week    : {self.base_week}",
            f"  Base Score   : {self.baseline_delivery:.2f}",
            f"  Breaking Point (score < 40): "
            + (f"+{self.breaking_point_pct:.1f}% order increase" if self.breaking_point_pct else "not reached"),
            "",
            f"  {'Scenario':<20}  {'Load':>7}  {'Delivery':>9}  {'Delta':>8}  Risk",
            "  " + "-" * 60,
        ]
        for s in self.scenarios:
            lines.append(
                f"  {s.label:<20}  +{s.order_increase_pct:>5.1f}%  "
                f"delivery={s.delivery_score:>6.2f}  "
                f"delta={s.delivery_delta:>+7.2f}  "
                f"risk={s.risk_level}"
            )
        return "\n".join(lines)


def _risk_level(delivery_score: float) -> str:
    if delivery_score >= 70: return "low"
    if delivery_score >= 55: return "moderate"
    if delivery_score >= 40: return "high"
    return "critical"


def _apply_load(base_score: float, increase_factor: float, elasticity: float = 1.0) -> float:
    return round(base_score / (increase_factor ** elasticity), 2)


def simulate_scenario(
    week: int,
    order_increase_pct: float,
    label: Optional[str] = None,
    picking_elasticity: float = 1.0,
    dispatch_elasticity: float = 1.2,
    df: Optional[pd.DataFrame] = None,
) -> ScenarioResult:
    m               = calculate_week_metrics(week, df=df)
    increase_factor = 1 + (order_increase_pct / 100)
    label           = label or f"+{order_increase_pct:.0f}% orders"

    new_picking  = _apply_load(m.picking_score,  increase_factor, picking_elasticity)
    new_dispatch = _apply_load(m.dispatch_score, increase_factor, dispatch_elasticity)
    new_delivery = round(
        WEIGHTS["picking"]  * new_picking +
        WEIGHTS["packing"]  * m.packing_score +
        WEIGHTS["dispatch"] * new_dispatch,
        2,
    )

    return ScenarioResult(
        label=label,
        order_increase_pct=order_increase_pct,
        picking_score=new_picking,
        packing_score=m.packing_score,
        dispatch_score=new_dispatch,
        delivery_score=new_delivery,
        delivery_delta=round(new_delivery - m.delivery_score, 2),
        risk_level=_risk_level(new_delivery),
    )


def simulate_load(
    week: int,
    order_increase_pct: float,
    df: Optional[pd.DataFrame] = None,
) -> ScenarioResult:
    return simulate_scenario(week, order_increase_pct, df=df)


def simulate_load_range(
    week: int,
    increments: list[float],
    df: Optional[pd.DataFrame] = None,
) -> LoadSimulationReport:
    base      = calculate_week_metrics(week, df=df)
    scenarios = [simulate_scenario(week, pct, df=df) for pct in increments]
    breaking  = next((s.order_increase_pct for s in scenarios if s.delivery_score < 40), None)

    return LoadSimulationReport(
        kpi="Delivery Timeliness",
        base_week=week,
        baseline_delivery=base.delivery_score,
        scenarios=scenarios,
        breaking_point_pct=breaking,
    )


def stress_test(
    week: int,
    step: float = 5.0,
    max_increase: float = 100.0,
    df: Optional[pd.DataFrame] = None,
) -> LoadSimulationReport:
    increments = [round(i, 1) for i in _frange(step, max_increase + step, step)]
    return simulate_load_range(week, increments, df=df)


def _frange(start: float, stop: float, step: float):
    val = start
    while val <= stop:
        yield val
        val += step


if __name__ == "__main__":
    from metrics import load_data
    df     = load_data()
    week   = sorted(df["week"].unique())[-1]
    report = stress_test(week, step=10.0, df=df)
    print(report.summary())