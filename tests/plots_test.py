# tests/plots_test.py
# from pathlib import Path

import matplotlib

from mlops.plots import PlotConfig, Plotter

matplotlib.use("Agg", force=True)


def test_plotter_line_creates_png(tmp_path):
    # Arrange: Plotter en modo script, guardando en carpeta temporal
    p = Plotter(modo="script", default_show=False, default_save=True, figures_dir=tmp_path)
    cfg = PlotConfig(title="test line", xlabel="x", ylabel="y")
    outfile = "line_test.png"

    # Act
    p.line([0, 1, 2], [0, 1, 4], cfg, save_as=outfile)

    # Assert
    outpath = tmp_path / outfile
    assert outpath.exists() and outpath.is_file()
    assert outpath.stat().st_size > 0


def test_plotter_bar_creates_png(tmp_path):
    # Arrange
    p = Plotter(modo="script", default_show=False, default_save=True, figures_dir=tmp_path)
    cfg = PlotConfig(title="test bar", xlabel="cat", ylabel="value")
    outfile = "bar_test.png"

    # Act
    p.bar(["A", "B", "C"], [3, 5, 2], cfg, save_as=outfile)

    # Assert
    outpath = tmp_path / outfile
    assert outpath.exists() and outpath.is_file()
    assert outpath.stat().st_size > 0
