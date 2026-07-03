from app.tools.charts import CreateChartTool


def test_bar_chart():
    tool = CreateChartTool()
    result = tool.run({
        "chart_type": "bar",
        "title": "营收对比",
        "data": [{"label": "2024", "values": [100, 200, 300]}],
        "x_labels": ["Q1", "Q2", "Q3"],
    })
    assert "chart_option" in result
    assert result["chart_type"] == "bar"
    option = result["chart_option"]
    assert option["title"]["text"] == "营收对比"
    assert option["series"][0]["type"] == "bar"
    assert option["xAxis"]["data"] == ["Q1", "Q2", "Q3"]


def test_line_chart():
    tool = CreateChartTool()
    result = tool.run({
        "chart_type": "line",
        "title": "趋势",
        "data": [{"label": "收入", "values": [1, 2, 3]}, {"label": "成本", "values": [0.5, 1, 1.5]}],
        "x_labels": ["2022", "2023", "2024"],
    })
    assert result["chart_type"] == "line"
    assert len(result["chart_option"]["series"]) == 2


def test_pie_chart():
    tool = CreateChartTool()
    result = tool.run({
        "chart_type": "pie",
        "title": "占比",
        "data": [{"label": "A", "values": [30, 50, 20]}],
        "x_labels": ["业务A", "业务B", "业务C"],
    })
    option = result["chart_option"]
    assert option["series"][0]["type"] == "pie"
    assert len(option["series"][0]["data"]) == 3


def test_scatter_chart():
    tool = CreateChartTool()
    result = tool.run({
        "chart_type": "scatter",
        "title": "散点",
        "data": [
            {"label": "x", "values": [1, 2, 3]},
            {"label": "y", "values": [4, 5, 6]},
        ],
    })
    assert result["chart_type"] == "scatter"
    assert result["chart_option"]["series"][0]["type"] == "scatter"


def test_missing_required_fields():
    tool = CreateChartTool()
    result = tool.run({"chart_type": "bar"})
    assert "error" in result


def test_unknown_chart_type():
    tool = CreateChartTool()
    result = tool.run({
        "chart_type": "heatmap",
        "title": "test",
        "data": [{"label": "a", "values": [1]}],
    })
    assert "error" in result
