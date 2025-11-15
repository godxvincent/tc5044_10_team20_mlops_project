# mlops/plots.py
"""
Módulo para generar gráficos de forma flexible.
Detecta si se ejecuta en notebook o terminal y actúa en consecuencia.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt


def _is_notebook() -> bool:
    """Detecta si el entorno actual es un notebook."""
    try:
        from IPython import get_ipython  # type: ignore

        shell = get_ipython().__class__.__name__  # noqa: SLF001
        return shell in {"ZMQInteractiveShell", "Shell"}  # lab/notebook
    except Exception:
        return False


@dataclass
class PlotConfig:
    """Configuración básica de los gráficos."""

    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    grid: bool = True
    tight_layout: bool = True


class Plotter:
    """
    Clase que gestiona la creación y visualización de gráficos
    con detección automática de entorno (notebook o script).
    """

    def __init__(
        self,
        modo: str = "auto",
        figures_dir: str | Path = "reports/figures",
        default_show: bool = True,
        default_save: bool = False,
    ) -> None:
        """Inicializa el Plotter y configura el backend apropiado."""
        env_mode = os.getenv("PLOTTER_MODE")
        self.modo = (env_mode or modo).lower()
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.default_show = default_show
        self.default_save = default_save

        # Detecta entorno si está en modo automático
        if self.modo == "auto":
            self.modo = "notebook" if _is_notebook() else "script"

        # 🧩 Fuerza backend sin GUI cuando se ejecuta en modo 'script'
        # Esto evita errores con Tkinter o entornos sin interfaz gráfica
        if self.modo == "script":
            matplotlib.use("Agg", force=True)

    # ----------------- Gráficos principales -----------------

    def line(
        self,
        x,
        y,
        cfg: Optional[PlotConfig] = None,
        *,
        save_as: Optional[str] = None,
        show: Optional[bool] = None,
        save: Optional[bool] = None,
        style: Optional[dict] = None,
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Genera una gráfica de líneas."""
        fig, ax = plt.subplots()
        ax.plot(x, y, **(style or {}))
        self._apply_cfg(ax, cfg)
        return self._finalize(fig, ax, save_as=save_as, show=show, save=save)

    def bar(
        self,
        labels,
        values,
        cfg: Optional[PlotConfig] = None,
        *,
        save_as: Optional[str] = None,
        show: Optional[bool] = None,
        save: Optional[bool] = None,
        style: Optional[dict] = None,
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Genera una gráfica de barras."""
        fig, ax = plt.subplots()
        ax.bar(labels, values, **(style or {}))
        self._apply_cfg(ax, cfg)
        return self._finalize(fig, ax, save_as=save_as, show=show, save=save)

    # ----------------- Métodos internos -----------------

    @staticmethod
    def _apply_cfg(ax: matplotlib.axes.Axes, cfg: Optional[PlotConfig]) -> None:
        cfg = cfg or PlotConfig()
        if cfg.title:
            ax.set_title(cfg.title)
        if cfg.xlabel:
            ax.set_xlabel(cfg.xlabel)
        if cfg.ylabel:
            ax.set_ylabel(cfg.ylabel)
        if cfg.grid:
            ax.grid(True)
        if cfg.tight_layout:
            plt.tight_layout()

    def _finalize(
        self,
        fig: matplotlib.figure.Figure,
        ax: matplotlib.axes.Axes,
        *,
        save_as: Optional[str],
        show: Optional[bool],
        save: Optional[bool],
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        do_show = self.default_show if show is None else show
        do_save = self.default_save if save is None else save

        if self.modo == "notebook":
            if do_show:
                plt.show()
            if do_save and save_as:
                fig.savefig(self.figures_dir / save_as, bbox_inches="tight")
        else:
            if do_save and save_as:
                fig.savefig(self.figures_dir / save_as, bbox_inches="tight")
            if do_show:
                try:
                    plt.show()
                except Exception:
                    if not do_save and save_as:
                        fig.savefig(self.figures_dir / save_as, bbox_inches="tight")
        return fig, ax


# ---------------------- Ejemplo: ----------------------

if __name__ == "__main__":
    plotter = Plotter(modo="auto")
    cfg = PlotConfig(title="Ejemplo de Línea", xlabel="X", ylabel="Y")
    plotter.line(range(10), [x**2 for x in range(10)], cfg, save_as="line_example.png", save=True)
    print("Gráfico generado en reports/figures/line_example.png")
