"""Visualization utilities for ZeroWaste-Pattern environment.

Provides matplotlib-based rendering of fabric layouts, piece placements,
grain lines, and performance metrics over time.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
from shapely.geometry import Polygon

from models.state import EnvironmentState


# Hatch patterns for pieces when colors overlap
HATCH_PATTERNS = ["", "/", "\\", "x", ".", "*", "o", "+", "-", "|"]


class FabricVisualizer:
    """Visualizer for the ZeroWaste-Pattern fabric environment.

    Renders fabric layouts with placed pieces, grain line indicators,
    occupancy grids, and performance metrics.

    Attributes:
        env_state: Current environment state to visualize.
        fabric_width: Width of the fabric in cm.
        fabric_length: Length of the fabric to display in cm.
    """

    def __init__(
        self,
        env_state: EnvironmentState,
        fabric_width: float,
        fabric_length: float,
    ) -> None:
        """Initialize the visualizer with environment state and fabric dimensions.

        Args:
            env_state: The EnvironmentState to render.
            fabric_width: Width of the fabric in cm.
            fabric_length: Maximum/current length of the fabric in cm.
        """
        self.env_state = env_state
        self.fabric_width = fabric_width
        self.fabric_length = fabric_length

    def render(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
    ) -> Figure:
        """Render the current fabric layout with all placed pieces.

        Draws the fabric rectangle, placed polygons with distinct colors,
        piece labels, grain line arrows, and utilization stats in the title.

        Args:
            save_path: If provided, save the figure to this file path.
            show: If True, display the figure interactively.
            figsize: Figure size as (width, height) in inches.
            title: Custom title for the figure.

        Returns:
            The matplotlib Figure object.
        """
        aspect = self.fabric_length / max(self.fabric_width, 1e-6)

        if figsize is None:
            fig_w = 10
            fig_h = max(8, min(20, int(fig_w * aspect * 0.8)))
            figsize = (fig_w, fig_h)

        fig, ax = plt.subplots(figsize=figsize)

        # Draw fabric background
        fabric_rect = plt.Rectangle(
            (0, 0),
            self.fabric_width,
            self.fabric_length,
            linewidth=2,
            edgecolor="black",
            facecolor="#F5F5DC",  # beige = fabric color
            zorder=0,
        )
        ax.add_patch(fabric_rect)

        # Draw subtle grid lines
        grid_spacing = 20.0
        for x in np.arange(0, self.fabric_width, grid_spacing):
            ax.axvline(x=x, color="#DDDDDD", linewidth=0.3, zorder=1)
        for y in np.arange(0, self.fabric_length, grid_spacing):
            ax.axhline(y=y, color="#DDDDDD", linewidth=0.3, zorder=1)

        # Draw placed pieces
        for idx, placed_piece in enumerate(self.env_state.placed_pieces):
            poly = placed_piece.get_polygon()
            if poly.is_empty:
                continue

            color = placed_piece.piece.color or f"C{idx % 10}"
            hatch = HATCH_PATTERNS[idx % len(HATCH_PATTERNS)]

            self._draw_polygon(
                ax=ax,
                polygon=poly,
                color=color,
                hatch=hatch,
                label=placed_piece.piece.name,
                alpha=0.7,
                zorder=2 + idx,
            )

            # Draw grain line arrow
            self._draw_grain_arrow(
                ax=ax,
                polygon=poly,
                grain_direction=placed_piece.piece.grain_direction,
                rotation_deg=placed_piece.rotation_deg,
                color=color,
                zorder=100 + idx,
            )

        # Draw fabric used length marker
        if self.env_state.fabric_length_used > 0:
            ax.axhline(
                y=self.env_state.fabric_length_used,
                color="red",
                linewidth=1.5,
                linestyle="--",
                label=f"Used length: {self.env_state.fabric_length_used:.1f} cm",
                zorder=200,
            )

        # Setup axes
        ax.set_xlim(-5, self.fabric_width + 5)
        ax.set_ylim(-5, self.fabric_length + 5)
        ax.set_xlabel("Width (cm)", fontsize=11)
        ax.set_ylabel("Length (cm)", fontsize=11)
        ax.set_aspect("equal")

        # Title with metrics
        if title is None:
            title = (
                f"ZeroWaste-Pattern Fabric Layout\n"
                f"Utilization: {self.env_state.utilization_pct:.1f}%  |  "
                f"Pieces Placed: {len(self.env_state.placed_pieces)}  |  "
                f"Step: {self.env_state.step_count}  |  "
                f"Invalid Actions: {self.env_state.invalid_action_count}"
            )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

        # Legend for placed pieces
        legend_patches = []
        seen_names = set()
        for idx, pp in enumerate(self.env_state.placed_pieces):
            name = pp.piece.name
            if name not in seen_names:
                color = pp.piece.color or f"C{idx % 10}"
                patch = mpatches.Patch(facecolor=color, alpha=0.7, label=name)
                legend_patches.append(patch)
                seen_names.add(name)

        if legend_patches:
            ax.legend(
                handles=legend_patches,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                fontsize=8,
                title="Pieces",
            )

        # Add utilization bar on the right side
        self._add_utilization_bar(fig, ax)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def _draw_polygon(
        self,
        ax: Axes,
        polygon: Polygon,
        color: str,
        hatch: str,
        label: str,
        alpha: float = 0.7,
        zorder: int = 2,
    ) -> None:
        """Draw a single Shapely polygon as a matplotlib patch.

        Args:
            ax: The matplotlib Axes to draw on.
            polygon: The Shapely Polygon to draw.
            color: Fill color string.
            hatch: Hatch pattern string.
            label: Label text to display at centroid.
            alpha: Fill transparency.
            zorder: Drawing order.
        """
        if polygon.is_empty:
            return

        # Extract exterior coordinates
        x, y = polygon.exterior.xy
        x_arr = np.array(x)
        y_arr = np.array(y)

        ax.fill(x_arr, y_arr, alpha=alpha, color=color, hatch=hatch, zorder=zorder)
        ax.plot(x_arr, y_arr, color="black", linewidth=0.8, zorder=zorder + 0.5)

        # Draw label at centroid
        centroid = polygon.centroid
        ax.text(
            centroid.x,
            centroid.y,
            label[:12],  # truncate long names
            fontsize=6,
            ha="center",
            va="center",
            fontweight="bold",
            color="black",
            zorder=zorder + 1,
        )

    def _draw_grain_arrow(
        self,
        ax: Axes,
        polygon: Polygon,
        grain_direction: Any,
        rotation_deg: float,
        color: str,
        zorder: int = 100,
    ) -> None:
        """Draw a grain line arrow on a placed piece.

        Args:
            ax: The matplotlib Axes to draw on.
            polygon: The placed polygon.
            grain_direction: The GrainDirection enum value.
            rotation_deg: The rotation applied during placement.
            color: Arrow color.
            zorder: Drawing order.
        """
        from models.pattern_piece import GrainDirection
        from utils.geometry import get_grain_angle

        if grain_direction == GrainDirection.ANY:
            return

        centroid = polygon.centroid
        minx, miny, maxx, maxy = polygon.bounds
        arrow_len = min(maxx - minx, maxy - miny) * 0.3

        if arrow_len < 2.0:
            return

        base_angle = get_grain_angle(grain_direction)
        # After rotation, grain angle = base_angle + rotation_deg
        # But we display in the placed coordinate frame
        effective_angle_rad = math.radians(base_angle + rotation_deg)

        dx = arrow_len * math.cos(effective_angle_rad)
        dy = arrow_len * math.sin(effective_angle_rad)

        ax.annotate(
            "",
            xy=(centroid.x + dx, centroid.y + dy),
            xytext=(centroid.x - dx, centroid.y - dy),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=1.5,
            ),
            zorder=zorder,
        )

    def _add_utilization_bar(self, fig: Figure, ax: Axes) -> None:
        """Add a small utilization percentage bar indicator to the figure.

        Args:
            fig: The matplotlib Figure.
            ax: The main Axes.
        """
        utilization = self.env_state.utilization_pct / 100.0
        waste = 1.0 - utilization

        # Add a small annotation box
        stats_text = (
            f"Utilization: {self.env_state.utilization_pct:.1f}%\n"
            f"Waste: {waste * 100:.1f}%\n"
            f"Fabric used: {self.env_state.fabric_length_used:.1f} cm"
        )
        ax.text(
            0.02,
            0.02,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            zorder=300,
        )

    def render_step_by_step(
        self,
        states: List[EnvironmentState],
        save_dir: Optional[str] = None,
        show: bool = False,
    ) -> List[Figure]:
        """Render a sequence of environment states as individual figures.

        Args:
            states: List of EnvironmentState snapshots.
            save_dir: If provided, save each figure to this directory.
            show: If True, display each figure interactively.

        Returns:
            List of matplotlib Figure objects, one per state.
        """
        figures = []
        for step_idx, state in enumerate(states):
            viz = FabricVisualizer(state, self.fabric_width, self.fabric_length)
            save_path = None
            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"step_{step_idx:04d}.png")

            fig = viz.render(
                save_path=save_path,
                show=show,
                title=f"Step {step_idx} | Utilization: {state.utilization_pct:.1f}%",
            )
            figures.append(fig)

        return figures

    def plot_metrics(
        self,
        metrics_history: List[Dict[str, Any]],
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Figure:
        """Plot performance metrics over the course of an episode.

        Creates a multi-panel plot showing utilization, cumulative reward,
        and invalid actions over episode steps.

        Args:
            metrics_history: List of per-step metric dictionaries. Each dict
                should have keys: "utilization", "reward", "valid".
            save_path: Optional path to save the figure.
            show: If True, display the figure interactively.

        Returns:
            The matplotlib Figure object.
        """
        if not metrics_history:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No metrics to display", ha="center", va="center")
            return fig

        steps = list(range(len(metrics_history)))
        utilizations = [h.get("utilization", 0.0) * 100.0 for h in metrics_history]
        rewards = [h.get("reward", 0.0) for h in metrics_history]
        cum_rewards = np.cumsum(rewards)
        invalid_flags = [0 if h.get("valid", True) else 1 for h in metrics_history]
        invalid_cumsum = np.cumsum(invalid_flags)

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Plot 1: Utilization over steps
        axes[0].plot(steps, utilizations, color="#2ECC71", linewidth=2, label="Utilization %")
        axes[0].set_ylabel("Utilization (%)", fontsize=10)
        axes[0].set_title("Fabric Utilization Over Episode", fontsize=11)
        axes[0].set_ylim(0, 105)
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)
        axes[0].fill_between(steps, utilizations, alpha=0.2, color="#2ECC71")

        # Plot 2: Cumulative reward
        axes[1].plot(steps, cum_rewards, color="#3498DB", linewidth=2, label="Cumulative Reward")
        axes[1].axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
        axes[1].set_ylabel("Cumulative Reward", fontsize=10)
        axes[1].set_title("Cumulative Reward Over Episode", fontsize=11)
        axes[1].legend(loc="upper left")
        axes[1].grid(True, alpha=0.3)

        # Shade positive/negative areas
        axes[1].fill_between(
            steps,
            cum_rewards,
            where=cum_rewards >= 0,
            alpha=0.2,
            color="#2ECC71",
            label="Positive",
        )
        axes[1].fill_between(
            steps,
            cum_rewards,
            where=cum_rewards < 0,
            alpha=0.2,
            color="#E74C3C",
            label="Negative",
        )

        # Plot 3: Cumulative invalid actions
        axes[2].plot(
            steps, invalid_cumsum, color="#E74C3C", linewidth=2, label="Cumulative Invalid Actions"
        )
        axes[2].set_ylabel("Invalid Actions (cumulative)", fontsize=10)
        axes[2].set_xlabel("Step", fontsize=10)
        axes[2].set_title("Invalid Action Count", fontsize=11)
        axes[2].legend(loc="upper left")
        axes[2].grid(True, alpha=0.3)

        # Mark individual invalid action events
        invalid_steps = [s for s, flag in enumerate(invalid_flags) if flag == 1]
        if invalid_steps:
            axes[2].scatter(
                invalid_steps,
                [invalid_cumsum[s] for s in invalid_steps],
                color="#E74C3C",
                s=20,
                zorder=5,
                alpha=0.7,
            )

        plt.suptitle(
            "ZeroWaste-Pattern: Episode Performance Metrics",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig


def render_occupancy_grid(
    occupancy_grid: np.ndarray,
    cell_size: float,
    fabric_width: float,
    fabric_length: float,
    title: str = "Occupancy Grid",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Figure:
    """Render the raw occupancy grid as a heatmap.

    Args:
        occupancy_grid: Binary H x W numpy array.
        cell_size: Size of each grid cell in cm.
        fabric_width: Fabric width in cm.
        fabric_length: Fabric length in cm.
        title: Plot title.
        save_path: Optional save path.
        show: Whether to display the figure.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 12))

    im = ax.imshow(
        occupancy_grid,
        origin="lower",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        aspect="auto",
        extent=[0, fabric_width, 0, fabric_length],
    )
    plt.colorbar(im, ax=ax, label="Occupancy (1=occupied)")
    ax.set_xlabel("Width (cm)")
    ax.set_ylabel("Length (cm)")
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig
