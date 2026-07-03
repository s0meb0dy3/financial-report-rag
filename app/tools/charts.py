from typing import Any

from app.tools.types import ChatTool


class CreateChartTool(ChatTool):
    name = "create_chart"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "create_chart",
                "description": "生成数据图表（柱状图、折线图、饼图、散点图）。图表会在回答中直接展示。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chart_type": {
                            "type": "string",
                            "enum": ["bar", "line", "pie", "scatter"],
                            "description": "图表类型",
                        },
                        "title": {
                            "type": "string",
                            "description": "图表标题",
                        },
                        "data": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "values": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                    },
                                },
                                "required": ["label", "values"],
                            },
                            "description": "数据系列。每个 series 有一个 label 和一组 values。",
                        },
                        "x_labels": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "X 轴标签（柱状图/折线图/散点图需要）",
                        },
                        "y_label": {
                            "type": "string",
                            "description": "Y 轴标签",
                        },
                        "width": {
                            "type": "number",
                            "description": "图表宽度（像素），默认 800",
                        },
                        "height": {
                            "type": "number",
                            "description": "图表高度（像素），默认 500",
                        },
                    },
                    "required": ["chart_type", "title", "data"],
                },
            },
        }

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        chart_type = arguments.get("chart_type")
        title = arguments.get("title", "")
        data = arguments.get("data", [])
        x_labels = arguments.get("x_labels", [])
        y_label = arguments.get("y_label", "")

        if not chart_type or not title or not data:
            return {"error": "chart_type, title, data 为必填字段"}

        builder = _ChartOptionBuilder(title=title, y_label=y_label)

        if chart_type == "bar":
            option = builder.bar(data, x_labels)
        elif chart_type == "line":
            option = builder.line(data, x_labels)
        elif chart_type == "pie":
            option = builder.pie(data, x_labels)
        elif chart_type == "scatter":
            option = builder.scatter(data, x_labels)
        else:
            return {"error": f"不支持的图表类型: {chart_type}"}

        return {"chart_option": option, "chart_type": chart_type}


class _ChartOptionBuilder:
    def __init__(self, *, title: str, y_label: str):
        self._title = title
        self._y_label = y_label

    def _base(self, series: list[dict], legend_names: list[str]) -> dict[str, Any]:
        option: dict[str, Any] = {
            "title": {"text": self._title, "left": "center"},
            "tooltip": {"trigger": "axis" if len(series) > 1 else "item"},
            "grid": {"containLabel": True},
        }
        if len(legend_names) > 1:
            option["legend"] = {"data": legend_names, "bottom": 0}
        return option

    def bar(self, data: list[dict], x_labels: list[str]) -> dict[str, Any]:
        series = [
            {"name": s["label"], "type": "bar", "data": s["values"]}
            for s in data
        ]
        option = self._base(series, [s["label"] for s in data])
        option["xAxis"] = {"type": "category", "data": x_labels}
        option["yAxis"] = {"type": "value", "name": self._y_label}
        option["series"] = series
        return option

    def line(self, data: list[dict], x_labels: list[str]) -> dict[str, Any]:
        series = [
            {"name": s["label"], "type": "line", "data": s["values"], "smooth": True}
            for s in data
        ]
        option = self._base(series, [s["label"] for s in data])
        option["xAxis"] = {"type": "category", "data": x_labels}
        option["yAxis"] = {"type": "value", "name": self._y_label}
        option["series"] = series
        return option

    def pie(self, data: list[dict], x_labels: list[str]) -> dict[str, Any]:
        values = data[0]["values"] if data else []
        pie_data = [
            {"name": name, "value": val}
            for name, val in zip(x_labels, values)
        ]
        option = self._base([], [])
        option["tooltip"] = {"trigger": "item"}
        option["series"] = [
            {
                "type": "pie",
                "radius": "55%",
                "center": ["50%", "50%"],
                "data": pie_data,
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowOffsetX": 0,
                        "shadowColor": "rgba(0, 0, 0, 0.5)",
                    }
                },
            }
        ]
        return option

    def scatter(self, data: list[dict], x_labels: list[str]) -> dict[str, Any]:
        x_vals = data[0]["values"] if len(data) >= 1 else []
        y_vals = data[1]["values"] if len(data) >= 2 else []
        scatter_data = list(zip(x_vals, y_vals))

        x_name = data[0]["label"] if len(data) >= 1 else ""
        y_name = data[1]["label"] if len(data) >= 2 else self._y_label

        series = [{"type": "scatter", "data": scatter_data, "symbolSize": 10}]
        option = self._base(series, [])
        option["xAxis"] = {"type": "value", "name": x_name}
        option["yAxis"] = {"type": "value", "name": y_name}
        option["series"] = series
        return option
