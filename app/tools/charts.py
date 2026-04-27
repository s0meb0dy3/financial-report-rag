from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Literal

from app.tools.base import RegisteredTool, ToolSpec


ChartType = Literal["bar", "grouped_bar", "line", "combo", "pie"]
MAX_SERIES = 8
MAX_CATEGORIES = 24
SUPPORTED_CHART_TYPES = {"bar", "grouped_bar", "line", "combo", "pie"}
CHART_COLORS = [
    "#1f7a57",
    "#a26f24",
    "#2b5f8f",
    "#7a4b2f",
    "#536a3a",
    "#8a6d3b",
    "#426f72",
    "#9b4d3c",
]


def build_create_chart_tool() -> RegisteredTool:
    def handler(
        chart_type: ChartType,
        title: str,
        categories: list[str],
        series: list[dict[str, Any]],
        unit: str | None = None,
        source_notes: list[str] | None = None,
    ) -> dict[str, Any]:
        spec = _normalize_chart_input(
            chart_type=chart_type,
            title=title,
            categories=categories,
            series=series,
            unit=unit,
            source_notes=source_notes,
        )
        option = _build_echarts_option(spec)
        chart_id = _chart_id(spec)
        return {
            "chart_id": chart_id,
            "chart_type": spec["chart_type"],
            "title": spec["title"],
            "summary": _chart_summary(spec),
            "echarts_option": option,
            "source_notes": spec["source_notes"],
        }

    return RegisteredTool(
        spec=ToolSpec(
            name="create_chart",
            description=(
                "Create a safe ECharts option from already-verified financial data. "
                "Call search_reports first to obtain evidence; do not invent values."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "grouped_bar", "line", "combo", "pie"],
                        "description": "Chart style to generate.",
                    },
                    "title": {"type": "string", "description": "Chart title."},
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "X-axis categories, or pie slice names.",
                    },
                    "series": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "values": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                },
                                "type": {
                                    "type": ["string", "null"],
                                    "enum": ["bar", "line", None],
                                    "description": "Only used by combo charts.",
                                },
                                "unit": {"type": ["string", "null"]},
                                "y_axis": {
                                    "type": ["string", "null"],
                                    "enum": ["left", "right", None],
                                },
                            },
                            "required": ["name", "values"],
                            "additionalProperties": False,
                        },
                    },
                    "unit": {"type": ["string", "null"], "description": "Default display unit."},
                    "source_notes": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Evidence notes such as document name and page.",
                    },
                },
                "required": ["chart_type", "title", "categories", "series"],
                "additionalProperties": False,
            },
        ),
        handler=handler,
    )


def _normalize_chart_input(
    *,
    chart_type: str,
    title: str,
    categories: list[str],
    series: list[dict[str, Any]],
    unit: str | None,
    source_notes: list[str] | None,
) -> dict[str, Any]:
    if chart_type not in SUPPORTED_CHART_TYPES:
        raise ValueError(f"Unsupported chart_type: {chart_type}")
    normalized_title = _clean_text(title, "title")
    normalized_categories = [_clean_text(item, "category") for item in categories or []]
    if not normalized_categories:
        raise ValueError("categories must not be empty")
    if len(normalized_categories) > MAX_CATEGORIES:
        raise ValueError(f"categories must contain at most {MAX_CATEGORIES} items")
    if not series:
        raise ValueError("series must not be empty")
    if len(series) > MAX_SERIES:
        raise ValueError(f"series must contain at most {MAX_SERIES} items")

    normalized_series = [
        _normalize_series_item(item, len(normalized_categories), chart_type)
        for item in series
    ]
    if chart_type == "pie" and len(normalized_series) != 1:
        raise ValueError("pie charts require exactly one series")

    return {
        "chart_type": chart_type,
        "title": normalized_title,
        "categories": normalized_categories,
        "series": normalized_series,
        "unit": _optional_clean_text(unit),
        "source_notes": [_clean_text(item, "source note") for item in source_notes or []],
    }


def _normalize_series_item(
    item: dict[str, Any],
    category_count: int,
    chart_type: str,
) -> dict[str, Any]:
    name = _clean_text(item.get("name"), "series name")
    raw_values = item.get("values")
    if not isinstance(raw_values, list):
        raise ValueError("series values must be a list")
    if len(raw_values) != category_count:
        raise ValueError("series values length must match categories length")
    values = [_finite_number(value) for value in raw_values]

    raw_type = item.get("type")
    series_type = raw_type if raw_type in {"bar", "line"} else None
    if chart_type == "line":
        series_type = "line"
    elif chart_type in {"bar", "grouped_bar", "pie"}:
        series_type = "bar"
    elif chart_type == "combo":
        series_type = series_type or "bar"

    raw_y_axis = item.get("y_axis")
    y_axis = raw_y_axis if raw_y_axis in {"left", "right"} else "left"
    return {
        "name": name,
        "values": values,
        "type": series_type,
        "unit": _optional_clean_text(item.get("unit")),
        "y_axis": y_axis,
    }


def _build_echarts_option(spec: dict[str, Any]) -> dict[str, Any]:
    base: dict[str, Any] = {
        "color": CHART_COLORS,
        "title": {
            "text": spec["title"],
            "left": 0,
            "top": 0,
            "textStyle": {"fontSize": 16, "fontWeight": 650, "color": "#171717"},
        },
        "tooltip": {"trigger": "axis" if spec["chart_type"] != "pie" else "item"},
        "legend": {
            "top": 32,
            "left": 0,
            "textStyle": {"color": "#6f6a60"},
        },
        "grid": {"left": 44, "right": 24, "top": 84, "bottom": 42, "containLabel": True},
    }

    if spec["chart_type"] == "pie":
        values = spec["series"][0]["values"]
        base["series"] = [
            {
                "name": spec["series"][0]["name"],
                "type": "pie",
                "radius": ["42%", "68%"],
                "center": ["50%", "56%"],
                "avoidLabelOverlap": True,
                "label": {"color": "#171717"},
                "data": [
                    {"name": name, "value": value}
                    for name, value in zip(spec["categories"], values, strict=True)
                ],
            }
        ]
        base.pop("grid", None)
        return base

    base["xAxis"] = {
        "type": "category",
        "data": spec["categories"],
        "axisLabel": {"color": "#6f6a60"},
        "axisLine": {"lineStyle": {"color": "#dfd8cc"}},
    }
    base["yAxis"] = _build_y_axis(spec)
    base["series"] = [
        {
            "name": item["name"],
            "type": item["type"],
            "data": item["values"],
            "smooth": item["type"] == "line",
            "yAxisIndex": 1 if spec["chart_type"] == "combo" and item["y_axis"] == "right" else 0,
            "barMaxWidth": 42,
        }
        for item in spec["series"]
    ]
    return base


def _build_y_axis(spec: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
    unit = spec["unit"]
    axis_name = unit or ""
    base_axis = {
        "type": "value",
        "name": axis_name,
        "nameTextStyle": {"color": "#6f6a60"},
        "axisLabel": {"color": "#6f6a60"},
        "splitLine": {"lineStyle": {"color": "#ebe4d8"}},
    }
    if spec["chart_type"] != "combo":
        return base_axis
    right_unit = next(
        (item["unit"] for item in spec["series"] if item["y_axis"] == "right" and item["unit"]),
        None,
    )
    return [
        base_axis,
        {
            **base_axis,
            "name": right_unit or axis_name,
            "splitLine": {"show": False},
        },
    ]


def _chart_id(spec: dict[str, Any]) -> str:
    payload = json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"chart-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def _chart_summary(spec: dict[str, Any]) -> str:
    points = len(spec["categories"]) * len(spec["series"])
    return f"生成 {spec['chart_type']} 图表，{len(spec['series'])} 个系列，{points} 个数据点"


def _clean_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{label} must not be empty")
    return cleaned


def _optional_clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("optional text values must be strings")
    cleaned = value.strip()
    return cleaned or None


def _finite_number(value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("series values must be numbers")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("series values must be finite numbers")
    return number
