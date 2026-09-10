from tools.charts import plot_chart


def test_plot_chart_bar(raw_df):
    out = plot_chart(raw_df, type="bar", x="region", y="montant_total", titre="CA par région")
    assert out["status"] == "ok"
    assert "figure" in out["detail"]


def test_plot_chart_bad_type(raw_df):
    out = plot_chart(raw_df, type="camembert", x="region", y="montant_total")
    assert out["status"] == "error"
