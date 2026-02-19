import logging
from typing import Any, Dict

from Contextbuilder import build_context
from Llmclient import call_llm

logger = logging.getLogger(__name__)


def _fetch_metrics(week: int) -> Dict[str, Any]:
    base = {
        "delivery_performance": 91.5,
        "order_accuracy": 98.2,
        "inbound_time": 42,
        "picking_time": 18,
        "packing_time": 12,
        "dispatch_time": 25,
        "on_time_delivery": 88.0,
        "warehouse_utilization": 74.3,
    }
    offset = (week % 5) * 1.3
    return {k: round(v + offset, 1) if isinstance(v, float) else v + int(offset)
            for k, v in base.items()}


def _fetch_metric_tree(week: int) -> Dict[str, Any]:
    return {
        "name": "Supply Chain Performance",
        "value": "",
        "status": "normal",
        "children": [
            {
                "name": "Fulfilment",
                "value": "",
                "status": "normal",
                "children": [
                    {"name": "Picking Time",   "value": f"{18 + week % 3} mins", "status": "normal",  "children": []},
                    {"name": "Packing Time",   "value": f"{12 + week % 2} mins", "status": "normal",  "children": []},
                    {"name": "Dispatch Time",  "value": f"{25 + week % 4} mins", "status": "warning", "children": []},
                ],
            },
            {
                "name": "Delivery",
                "value": "",
                "status": "warning",
                "children": [
                    {"name": "On-Time Delivery",     "value": f"{88.0 - week % 3:.1f}%", "status": "warning", "children": []},
                    {"name": "Delivery Performance", "value": f"{91.5 + week % 2:.1f}%", "status": "normal",  "children": []},
                ],
            },
            {
                "name": "Warehouse",
                "value": "",
                "status": "normal",
                "children": [
                    {"name": "Inbound Time",          "value": f"{42 + week % 3} mins",   "status": "normal", "children": []},
                    {"name": "Warehouse Utilization", "value": f"{74.3 + week % 2:.1f}%", "status": "normal", "children": []},
                ],
            },
        ],
    }


def _fetch_root_cause(week: int) -> Dict[str, Any]:
    return {
        "primary_cause": "Dispatch time spike in Week {} due to carrier delays".format(week),
        "details": "Third-party carrier reported road network congestion affecting last-mile delivery.",
        "contributing_factors": [
            "Increased order volume (+12 % WoW)",
            "Reduced carrier fleet availability on weekends",
            "Manual sorting errors at dispatch hub",
        ],
    }


def _fetch_anomalies(week: int) -> Dict[str, Any]:
    return {
        "anomalies": [
            {
                "metric": "Dispatch Time",
                "severity": "warning",
                "description": "25 % above weekly average – breached SLA threshold.",
            },
            {
                "metric": "On-Time Delivery",
                "severity": "critical",
                "description": "Dropped below 90 % target for the second consecutive week.",
            },
        ]
    }


def generate_week_insights(current_week: int, previous_week: int) -> Dict[str, Any]:
    logger.info("Generating insights: current_week=%s, previous_week=%s", current_week, previous_week)

    try:
        current_metrics  = _fetch_metrics(current_week)
        previous_metrics = _fetch_metrics(previous_week)
        metric_tree      = _fetch_metric_tree(current_week)
        root_cause       = _fetch_root_cause(current_week)
        anomalies        = _fetch_anomalies(current_week)
    except Exception as exc:
        logger.exception("Failed to fetch operational data")
        return {
            "status": "Error",
            "summary": f"Data fetch error: {exc}",
            "bottleneck": None,
            "root_cause": None,
            "recommendations": [],
        }

    context = build_context(
        current_week=current_week,
        previous_week=previous_week,
        current_metrics=current_metrics,
        previous_metrics=previous_metrics,
        metric_tree=metric_tree,
        root_cause=root_cause,
        anomalies=anomalies,
    )

    logger.debug("Built context:\n%s", context)

    try:
        result = call_llm(context)
    except Exception as exc:
        logger.exception("LLM call failed")
        return {
            "status": "Error",
            "summary": f"LLM error: {exc}",
            "bottleneck": None,
            "root_cause": None,
            "recommendations": [],
        }

    result.setdefault("status", "Alert")
    result.setdefault("summary", "")
    result.setdefault("bottleneck", None)
    result.setdefault("root_cause", None)
    result.setdefault("recommendations", [])

    return result
