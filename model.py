from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from metrics import load_data, BENCHMARKS


FEATURES      = ["pick_time", "pack_time", "dispatch_delay", "error_count"]
CONTAMINATION = 0.15


@dataclass
class Anomaly:
    week: int
    row_index: int
    anomaly_score: float
    severity: str
    triggered_features: list[str]
    values: dict[str, float]
    explanation: str


@dataclass
class AnomalyReport:
    total_records: int
    anomaly_count: int
    anomaly_rate: float
    severity_breakdown: dict[str, int]
    anomalies: list[Anomaly]
    most_common_trigger: str

    def summary(self) -> str:
        lines = [
            "Anomaly Detection Report",
            f"  Total Records   : {self.total_records}",
            f"  Anomalies Found : {self.anomaly_count} ({self.anomaly_rate:.1%})",
            f"  Most Triggered  : {self.most_common_trigger}",
            "",
            "  Severity Breakdown:",
        ]
        for level, count in self.severity_breakdown.items():
            bar = "#" * count
            lines.append(f"    {level:<10} {count:>3}  |{bar}")

        lines += ["", f"  {'Week':<6} {'Severity':<10} {'Score':>7}  Triggers  Explanation"]
        lines.append("  " + "-" * 70)
        for a in self.anomalies:
            triggers = ", ".join(a.triggered_features) or "none"
            lines.append(
                f"  {a.week:<6} {a.severity:<10} {a.anomaly_score:>7.4f}  "
                f"{triggers:<30}  {a.explanation}"
            )
        return "\n".join(lines)


def _severity(score: float) -> str:
    if score < -0.15: return "critical"
    if score < -0.05: return "high"
    return "moderate"


def _find_triggered_features(row: pd.Series) -> list[str]:
    triggered = []
    checks = {
        "pick_time":      ("pick_time",      "poor"),
        "pack_time":      ("pack_time",      "poor"),
        "dispatch_delay": ("dispatch_delay", "poor"),
    }
    for col, (bench_key, tier) in checks.items():
        if col in row and row[col] > BENCHMARKS[bench_key][tier]:
            triggered.append(col)
    if "error_count" in row and row["error_count"] >= 3:
        triggered.append("error_count")
    return triggered


def _explain(triggered: list[str], severity: str) -> str:
    if not triggered:
        return f"Unusual multivariate pattern detected ({severity} severity)."
    labels = [f.replace("_", " ") for f in triggered]
    return f"{severity.title()} anomaly driven by: {', '.join(labels)}."


def detect_anomalies(
    df: Optional[pd.DataFrame] = None,
    contamination: float = CONTAMINATION,
    week_filter: Optional[int] = None,
) -> AnomalyReport:
    if df is None:
        df = load_data()

    if week_filter is not None:
        df = df[df["week"] == week_filter].copy()

    available      = [f for f in FEATURES if f in df.columns]
    feature_matrix = df[available].copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)

    model = IsolationForest(contamination=contamination, n_estimators=200, random_state=42)
    df    = df.copy()
    df["anomaly_flag"]  = model.fit_predict(scaled)
    df["anomaly_score"] = model.decision_function(scaled)

    anomaly_rows = df[df["anomaly_flag"] == -1].copy()

    anomalies = []
    for idx, row in anomaly_rows.iterrows():
        triggered = _find_triggered_features(row)
        score     = round(row["anomaly_score"], 6)
        severity  = _severity(score)
        anomalies.append(Anomaly(
            week=int(row["week"]) if "week" in row else -1,
            row_index=idx,
            anomaly_score=score,
            severity=severity,
            triggered_features=triggered,
            values={f: round(row[f], 2) for f in available if f in row},
            explanation=_explain(triggered, severity),
        ))

    anomalies.sort(key=lambda a: a.anomaly_score)

    severity_breakdown = {"critical": 0, "high": 0, "moderate": 0}
    for a in anomalies:
        severity_breakdown[a.severity] += 1

    all_triggers = [f for a in anomalies for f in a.triggered_features]
    most_common  = max(set(all_triggers), key=all_triggers.count) if all_triggers else "none"

    return AnomalyReport(
        total_records=len(df),
        anomaly_count=len(anomalies),
        anomaly_rate=len(anomalies) / len(df) if len(df) > 0 else 0.0,
        severity_breakdown=severity_breakdown,
        most_common_trigger=most_common,
        anomalies=anomalies,
    )


def detect_anomalies_by_week(
    df: Optional[pd.DataFrame] = None,
) -> dict[int, AnomalyReport]:
    if df is None:
        df = load_data()
    return {
        int(week): detect_anomalies(df=df, week_filter=int(week))
        for week in sorted(df["week"].unique())
    }


if __name__ == "__main__":
    df     = load_data()
    report = detect_anomalies(df=df)
    print(report.summary())

    print(f"\n{'='*55}")
    print("  Per-Week Anomaly Summary")
    print(f"{'='*55}")
    for week, r in detect_anomalies_by_week(df=df).items():
        print(f"  Week {week}:  {r.anomaly_count} anomalies  "
              f"(critical={r.severity_breakdown['critical']}, "
              f"high={r.severity_breakdown['high']})")