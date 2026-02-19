from dataclasses import dataclass
from typing import Optional
import pandas as pd

from metrics import calculate_week_metrics, compare_weeks, WEIGHTS

@dataclass
class ImpactFactor:
    metric: str
    previous: float
    current: float
    change: float
    weight: float
    weighted_impact: float
    direction: str


@dataclass
class RootCauseReport:
    kpi: str
    current_week: int
    previous_week: int
    previous_score: float
    current_score: float
    total_drop: float
    main_driver: str
    drivers: list[ImpactFactor]
    verdict: str

    def summary(self) -> str:
        lines = [
            f"Root Cause Report: {self.kpi}",
            f"  Week {self.previous_week} -> Week {self.current_week}",
            f"  Score: {self.previous_score} -> {self.current_score}  ({"+" if self.total_drop < 0 else "-"}{abs(self.total_drop):.2f})",
            f"  Main Driver: {self.main_driver}",
            "",
            "  Impact Breakdown:",
        ]
        for d in self.drivers:
            bar = "#" * int(abs(d.weighted_impact) / 2)
            lines.append(
                f"    {d.metric:<20}  change={d.change:+.2f}  "
                f"impact={d.weighted_impact:+.2f}  {d.direction:<11}  |{bar}"
            )
        lines += ["", f"  Verdict: {self.verdict}"]
        return "\n".join(lines)


def root_cause_analysis(
    current_week: int,
    previous_week: int,
    df: Optional[pd.DataFrame] = None,
) -> RootCauseReport:
    current  = calculate_week_metrics(current_week,  df=df)
    previous = calculate_week_metrics(previous_week, df=df)

    score_map = {
        "picking_score":  ("picking",  current.picking_score,  previous.picking_score),
        "packing_score":  ("packing",  current.packing_score,  previous.packing_score),
        "dispatch_score": ("dispatch", current.dispatch_score, previous.dispatch_score),
    }

    drivers = []
    for metric, (key, curr_val, prev_val) in score_map.items():
        change          = curr_val - prev_val
        weight          = WEIGHTS[key]
        weighted_impact = round(weight * change, 2)
        drivers.append(ImpactFactor(
            metric=metric,
            previous=prev_val,
            current=curr_val,
            change=round(change, 2),
            weight=weight,
            weighted_impact=weighted_impact,
            direction="improving" if change > 0 else "degrading" if change < 0 else "stable",
        ))

    drivers.sort(key=lambda x: x.weighted_impact)

    total_drop  = round(previous.delivery_score - current.delivery_score, 2)
    main_driver = drivers[0].metric

    verdict = _generate_verdict(total_drop, drivers, current.error_rate)

    return RootCauseReport(
        kpi="Delivery Timeliness",
        current_week=current_week,
        previous_week=previous_week,
        previous_score=previous.delivery_score,
        current_score=current.delivery_score,
        total_drop=total_drop,
        main_driver=main_driver,
        drivers=drivers,
        verdict=verdict,
    )


def _generate_verdict(drop: float, drivers: list[ImpactFactor], error_rate: float) -> str:
    if drop <= 0:
        return "Performance improved week-over-week. No corrective action required."

    worst        = drivers[0]
    severity     = "critically" if drop > 15 else "significantly" if drop > 8 else "moderately"
    metric_label = worst.metric.replace("_score", "").replace("_", " ")
    error_note   = f" High error rate ({error_rate:.2f}) may be compounding the issue." if error_rate > 0.8 else ""

    return (
        f"Delivery score dropped {severity} by {drop:.2f} points. "
        f"{metric_label.title()} was the primary driver (impact: {worst.weighted_impact:+.2f}).{error_note} "
        f"Investigate {metric_label} operations first."
    )


def multi_week_root_cause(weeks: list[int], df: Optional[pd.DataFrame] = None) -> list[RootCauseReport]:
    return [
        root_cause_analysis(weeks[i], weeks[i - 1], df=df)
        for i in range(1, len(weeks))
    ]


if __name__ == "__main__":
    from metrics import load_data

    df    = load_data()
    weeks = sorted(df["week"].unique())

    if len(weeks) < 2:
        print("Need at least 2 weeks of data.")
    else:
        report = root_cause_analysis(weeks[-1], weeks[-2], df=df)
        print(report.summary())

        if len(weeks) > 2:
            print(f"\n{'='*55}")
            print("  Multi-Week Trend")
            print(f"{'='*55}")
            for r in multi_week_root_cause(weeks, df=df):
                arrow = "v" if r.total_drop > 0 else "^"
                print(f"  Week {r.previous_week} -> {r.current_week}  {arrow} {abs(r.total_drop):.2f}  driver: {r.main_driver}")