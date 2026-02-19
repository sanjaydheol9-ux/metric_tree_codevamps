def _fmt(value, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value}{suffix}"


def _delta(current, previous, suffix: str = "") -> str:
    if current is None or previous is None:
        return ""
    diff = current - previous
    sign = "+" if diff >= 0 else ""
    return f" ({sign}{diff:.1f}{suffix} vs previous week)"


def build_context(
    current_week: int,
    previous_week: int,
    current_metrics: dict,
    previous_metrics: dict,
    metric_tree: dict,
    root_cause: dict,
    anomalies: dict,
) -> str:
    sections = []

    c = current_metrics or {}
    p = previous_metrics or {}

    kpi_lines = [
        f"  Current Week  : Week {current_week}",
        f"  Previous Week : Week {previous_week}",
        "",
    ]

    kpi_fields = [
        ("delivery_performance",  "Delivery Performance", "%"),
        ("order_accuracy",        "Order Accuracy",       "%"),
        ("inbound_time",          "Inbound Time",         " mins"),
        ("picking_time",          "Picking Time",         " mins"),
        ("packing_time",          "Packing Time",         " mins"),
        ("dispatch_time",         "Dispatch Time",        " mins"),
        ("on_time_delivery",      "On-Time Delivery",     "%"),
        ("warehouse_utilization", "Warehouse Utilization","%"),
    ]

    for key, label, suffix in kpi_fields:
        cur_val = c.get(key)
        prv_val = p.get(key)
        if cur_val is not None or prv_val is not None:
            line = (
                f"  {label:<25}: "
                f"Current={_fmt(cur_val, suffix)}  |  "
                f"Previous={_fmt(prv_val, suffix)}"
                f"{_delta(cur_val, prv_val, suffix)}"
            )
            kpi_lines.append(line)

    sections.append("=== KPI COMPARISON ===\n" + "\n".join(kpi_lines))

    tree_lines = []
    if metric_tree:
        def _flatten_tree(node, depth=0):
            indent = "  " * depth
            name   = node.get("name") or node.get("metric") or str(node)
            value  = node.get("value") or node.get("current_value", "")
            status = node.get("status") or node.get("health") or ""
            flag   = " ⚠" if str(status).lower() in ("warning", "critical", "alert", "bad") else ""
            tree_lines.append(f"{indent}- {name}: {value}{flag}")
            for child in node.get("children", []):
                _flatten_tree(child, depth + 1)

        if isinstance(metric_tree, dict):
            _flatten_tree(metric_tree)
        elif isinstance(metric_tree, list):
            for node in metric_tree:
                _flatten_tree(node)

    sections.append(
        "=== METRIC TREE (Hierarchy) ===\n"
        + ("\n".join(tree_lines) if tree_lines else "  No metric tree data available.")
    )

    rc_lines = []
    if root_cause:
        primary      = root_cause.get("primary_cause") or root_cause.get("root_cause") or root_cause.get("cause")
        contributing = root_cause.get("contributing_factors") or root_cause.get("factors") or []
        details      = root_cause.get("details") or root_cause.get("explanation") or root_cause.get("description")

        if primary:
            rc_lines.append(f"  Primary Cause     : {primary}")
        if details:
            rc_lines.append(f"  Details           : {details}")
        if contributing:
            rc_lines.append("  Contributing Factors:")
            for factor in (contributing if isinstance(contributing, list) else [contributing]):
                rc_lines.append(f"    - {factor}")

        skip_keys = {
            "primary_cause", "root_cause", "cause",
            "contributing_factors", "factors",
            "details", "explanation", "description",
        }
        for k, v in root_cause.items():
            if k not in skip_keys and v:
                rc_lines.append(f"  {k}: {v}")

    sections.append(
        "=== ROOT CAUSE ANALYSIS ===\n"
        + ("\n".join(rc_lines) if rc_lines else "  No root cause data available.")
    )

    anomaly_lines = []
    if anomalies:
        items = (
            anomalies.get("anomalies")
            or anomalies.get("items")
            or (anomalies if isinstance(anomalies, list) else [])
        )
        if isinstance(items, list) and items:
            for a in items:
                if isinstance(a, dict):
                    metric   = a.get("metric") or a.get("name") or "Unknown metric"
                    severity = a.get("severity") or a.get("level") or ""
                    desc     = a.get("description") or a.get("details") or a.get("message") or ""
                    anomaly_lines.append(f"  [{severity.upper() or 'ANOMALY'}] {metric}: {desc}")
                else:
                    anomaly_lines.append(f"  - {a}")
        elif isinstance(anomalies, dict) and not items:
            for k, v in anomalies.items():
                if v:
                    anomaly_lines.append(f"  {k}: {v}")

    sections.append(
        "=== DETECTED ANOMALIES ===\n"
        + ("\n".join(anomaly_lines) if anomaly_lines else "  No anomalies detected.")
    )

    return "\n\n".join(sections)
