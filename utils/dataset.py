"""Pattern dataset generator for ZeroWaste-Pattern environment.

Provides stochastic and deterministic generators for garment pattern pieces,
including simple rectangles, realistic shirt patterns, and random shapes.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from models.pattern_piece import GrainDirection, PatternPiece


# Color palette for distinct piece visualization
PIECE_COLORS = [
    "#E74C3C",  # red
    "#3498DB",  # blue
    "#2ECC71",  # green
    "#F39C12",  # orange
    "#9B59B6",  # purple
    "#1ABC9C",  # teal
    "#E67E22",  # dark orange
    "#2980B9",  # dark blue
    "#27AE60",  # dark green
    "#8E44AD",  # dark purple
    "#16A085",  # dark teal
    "#D35400",  # brown-orange
    "#C0392B",  # dark red
    "#2C3E50",  # dark grey
    "#F1C40F",  # yellow
]


class PatternDataset:
    """Generator class for garment pattern piece datasets.

    Provides methods to generate pattern sets for different difficulty levels
    and garment types, suitable for training and evaluating the RL environment.
    """

    @staticmethod
    def generate_basic_set(n_pieces: int = 12, seed: Optional[int] = None) -> List[PatternPiece]:
        """Generate a basic set of rectangular pattern pieces.

        Creates rectangles of varying sizes with no grain constraints,
        suitable for the BasicPackingTask.

        Args:
            n_pieces: Number of rectangular pieces to generate.
            seed: Random seed for reproducibility.

        Returns:
            List of PatternPiece instances (rectangles).
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Predefined sizes to ensure variety
        size_presets = [
            (40, 60),  # large panel
            (30, 50),  # medium panel
            (20, 40),  # small panel
            (15, 35),  # narrow piece
            (10, 30),  # strip
            (50, 25),  # wide piece
            (35, 45),  # square-ish
            (25, 55),  # tall piece
            (18, 28),  # small square
            (45, 20),  # wide strip
            (12, 48),  # narrow tall
            (38, 32),  # medium wide
            (22, 18),  # small rectangle
            (55, 15),  # banner piece
            (28, 42),  # portrait piece
        ]

        pieces = []
        for i in range(n_pieces):
            if i < len(size_presets):
                w, h = size_presets[i]
            else:
                w = random.randint(8, 55)
                h = random.randint(8, 60)

            # Add slight random variation
            w += random.uniform(-2, 2)
            h += random.uniform(-2, 2)
            w = max(5.0, w)
            h = max(5.0, h)

            vertices = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
            color = PIECE_COLORS[i % len(PIECE_COLORS)]

            piece = PatternPiece(
                id=f"rect_{i+1:02d}",
                name=f"Rectangle {i+1}",
                vertices=vertices,
                grain_direction=GrainDirection.ANY,
                grain_tolerance_deg=45.0,
                allowed_rotations=[0.0, 90.0, 180.0, 270.0],
                quantity=1,
                color=color,
            )
            pieces.append(piece)

        return pieces

    @staticmethod
    def generate_shirt_set() -> List[PatternPiece]:
        """Generate a realistic shirt pattern set with actual polygon shapes.

        Creates 8+ piece types representing a complete shirt pattern:
        front_bodice, back_bodice, sleeve (x2), collar, cuff (x2),
        front_panel, pocket, collar_stand, yoke.

        All coordinates are in cm, scaled for a size M men's shirt.

        Returns:
            List of PatternPiece instances with realistic shirt piece shapes.
        """
        pieces = []

        # ---- Front Bodice ----
        # Trapezoidal shape with shoulder and armhole curves
        front_bodice = PatternPiece(
            id="front_bodice",
            name="Front Bodice",
            vertices=[
                (0.0, 0.0),    # bottom left
                (40.0, 0.0),   # bottom right
                (42.0, 10.0),  # side seam curve out
                (44.0, 30.0),  # side seam mid
                (40.0, 55.0),  # side seam top right
                (32.0, 60.0),  # shoulder right
                (24.0, 58.0),  # shoulder mid-right
                (18.0, 52.0),  # neckline curve right
                (12.0, 52.0),  # neckline curve left
                (6.0, 58.0),   # shoulder mid-left
                (0.0, 60.0),   # shoulder left
                (-2.0, 55.0),  # armhole top
                (-4.0, 40.0),  # armhole curve
                (-3.0, 20.0),  # armhole bottom
                (0.0, 10.0),   # side seam lower
            ],
            grain_direction=GrainDirection.VERTICAL,
            grain_tolerance_deg=5.0,
            allowed_rotations=[0.0, 180.0],
            quantity=1,
            color=PIECE_COLORS[0],
        )
        pieces.append(front_bodice)

        # ---- Back Bodice ----
        # Similar to front but with back yoke seam and no front opening
        back_bodice = PatternPiece(
            id="back_bodice",
            name="Back Bodice",
            vertices=[
                (0.0, 0.0),    # bottom left
                (42.0, 0.0),   # bottom right
                (44.0, 10.0),  # side seam lower right
                (46.0, 30.0),  # side seam mid
                (43.0, 55.0),  # side seam upper right
                (34.0, 62.0),  # shoulder right
                (25.0, 63.0),  # shoulder mid
                (10.0, 62.0),  # back neck
                (0.0, 62.0),   # center back top
                (-2.0, 58.0),  # shoulder left
                (-4.0, 40.0),  # armhole
                (-3.0, 20.0),  # armhole lower
                (0.0, 10.0),   # side seam lower left
            ],
            grain_direction=GrainDirection.VERTICAL,
            grain_tolerance_deg=5.0,
            allowed_rotations=[0.0, 180.0],
            quantity=1,
            color=PIECE_COLORS[1],
        )
        pieces.append(back_bodice)

        # ---- Sleeve ----
        # Classic shirt sleeve shape with cap curve
        sleeve = PatternPiece(
            id="sleeve_left",
            name="Left Sleeve",
            vertices=[
                (0.0, 0.0),    # cuff bottom left
                (20.0, 0.0),   # cuff bottom right
                (22.0, 5.0),   # cuff seam right
                (25.0, 15.0),  # lower sleeve right
                (26.0, 30.0),  # mid sleeve right
                (24.0, 45.0),  # upper sleeve right
                (20.0, 55.0),  # sleeve cap right start
                (16.0, 60.0),  # sleeve cap right curve
                (10.0, 63.0),  # sleeve cap top
                (4.0, 60.0),   # sleeve cap left curve
                (0.0, 55.0),   # sleeve cap left start
                (-4.0, 45.0),  # upper sleeve left
                (-5.0, 30.0),  # mid sleeve left
                (-3.0, 15.0),  # lower sleeve left
                (-1.0, 5.0),   # cuff seam left
            ],
            grain_direction=GrainDirection.VERTICAL,
            grain_tolerance_deg=5.0,
            allowed_rotations=[0.0, 180.0],
            quantity=2,
            color=PIECE_COLORS[2],
        )
        pieces.append(sleeve)

        # ---- Collar ----
        # Curved collar shape
        collar = PatternPiece(
            id="collar",
            name="Collar",
            vertices=[
                (0.0, 0.0),    # left bottom
                (30.0, 0.0),   # right bottom
                (32.0, 3.0),   # right lower edge
                (33.0, 7.0),   # right tip
                (30.0, 12.0),  # right outer edge
                (20.0, 13.0),  # right collar point
                (15.0, 12.0),  # collar center top
                (10.0, 13.0),  # left collar point
                (0.0, 12.0),   # left outer edge
                (-3.0, 7.0),   # left tip
                (-2.0, 3.0),   # left lower edge
            ],
            grain_direction=GrainDirection.HORIZONTAL,
            grain_tolerance_deg=5.0,
            allowed_rotations=[0.0, 180.0],
            quantity=2,  # upper and under collar
            color=PIECE_COLORS[3],
        )
        pieces.append(collar)

        # ---- Collar Stand ----
        # Rectangular-ish piece that attaches collar to neckline
        collar_stand = PatternPiece(
            id="collar_stand",
            name="Collar Stand",
            vertices=[
                (0.0, 0.0),
                (32.0, 0.0),
                (33.0, 2.0),
                (33.5, 5.0),
                (33.0, 8.0),
                (32.0, 8.0),
                (0.0, 8.0),
                (-1.0, 5.0),
                (-0.5, 2.0),
            ],
            grain_direction=GrainDirection.HORIZONTAL,
            grain_tolerance_deg=5.0,
            allowed_rotations=[0.0, 180.0],
            quantity=2,
            color=PIECE_COLORS[4],
        )
        pieces.append(collar_stand)

        # ---- Cuff ----
        # Rectangular cuff piece
        cuff = PatternPiece(
            id="cuff",
            name="Cuff",
            vertices=[
                (0.0, 0.0),
                (25.0, 0.0),
                (26.0, 2.0),
                (26.0, 8.0),
                (25.0, 10.0),
                (0.0, 10.0),
                (-1.0, 8.0),
                (-1.0, 2.0),
            ],
            grain_direction=GrainDirection.HORIZONTAL,
            grain_tolerance_deg=5.0,
            allowed_rotations=[0.0, 90.0, 180.0, 270.0],
            quantity=2,
            color=PIECE_COLORS[5],
        )
        pieces.append(cuff)

        # ---- Front Placket / Button Band ----
        front_panel = PatternPiece(
            id="front_panel",
            name="Front Button Band",
            vertices=[
                (0.0, 0.0),
                (5.0, 0.0),
                (5.5, 5.0),
                (6.0, 20.0),
                (5.5, 40.0),
                (5.0, 55.0),
                (4.5, 60.0),
                (0.0, 60.0),
                (-0.5, 55.0),
                (-1.0, 40.0),
                (-0.5, 20.0),
                (0.0, 5.0),
            ],
            grain_direction=GrainDirection.VERTICAL,
            grain_tolerance_deg=5.0,
            allowed_rotations=[0.0, 180.0],
            quantity=2,
            color=PIECE_COLORS[6],
        )
        pieces.append(front_panel)

        # ---- Pocket ----
        # Small chest pocket
        pocket = PatternPiece(
            id="pocket",
            name="Chest Pocket",
            vertices=[
                (0.0, 0.0),
                (12.0, 0.0),
                (13.0, 1.0),
                (13.0, 13.0),
                (12.5, 15.0),
                (6.5, 16.0),
                (6.0, 16.0),
                (0.0, 15.0),
                (-0.5, 13.0),
                (-0.5, 1.0),
            ],
            grain_direction=GrainDirection.VERTICAL,
            grain_tolerance_deg=5.0,
            allowed_rotations=[0.0, 90.0, 180.0, 270.0],
            quantity=1,
            color=PIECE_COLORS[7],
        )
        pieces.append(pocket)

        # ---- Back Yoke ----
        # Horizontal back yoke piece
        back_yoke = PatternPiece(
            id="back_yoke",
            name="Back Yoke",
            vertices=[
                (0.0, 0.0),
                (42.0, 0.0),
                (44.0, 3.0),
                (44.0, 15.0),
                (42.0, 18.0),
                (30.0, 20.0),
                (15.0, 20.0),
                (0.0, 18.0),
                (-2.0, 15.0),
                (-2.0, 3.0),
            ],
            grain_direction=GrainDirection.HORIZONTAL,
            grain_tolerance_deg=5.0,
            allowed_rotations=[0.0, 180.0],
            quantity=2,  # outer and lining
            color=PIECE_COLORS[8],
        )
        pieces.append(back_yoke)

        # ---- Side Panel ----
        # Optional side panel for fitted shirt
        side_panel = PatternPiece(
            id="side_panel",
            name="Side Panel",
            vertices=[
                (0.0, 0.0),
                (12.0, 0.0),
                (14.0, 10.0),
                (15.0, 30.0),
                (13.0, 50.0),
                (10.0, 58.0),
                (0.0, 60.0),
                (-3.0, 50.0),
                (-4.0, 30.0),
                (-3.0, 10.0),
            ],
            grain_direction=GrainDirection.VERTICAL,
            grain_tolerance_deg=5.0,
            allowed_rotations=[0.0, 180.0],
            quantity=2,
            color=PIECE_COLORS[9],
        )
        pieces.append(side_panel)

        return pieces

    @staticmethod
    def generate_random_set(
        n_pieces: int = 10,
        complexity: str = "medium",
        seed: Optional[int] = None,
    ) -> List[PatternPiece]:
        """Generate a random set of pattern pieces with varying complexity.

        Args:
            n_pieces: Number of pieces to generate.
            complexity: One of "low", "medium", "high". Controls polygon complexity.
            seed: Random seed for reproducibility.

        Returns:
            List of randomly generated PatternPiece instances.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        complexity_params = {
            "low": {"min_verts": 4, "max_verts": 6, "irregularity": 0.1},
            "medium": {"min_verts": 5, "max_verts": 8, "irregularity": 0.25},
            "high": {"min_verts": 6, "max_verts": 12, "irregularity": 0.4},
        }
        params = complexity_params.get(complexity, complexity_params["medium"])

        pieces = []
        for i in range(n_pieces):
            n_verts = random.randint(params["min_verts"], params["max_verts"])
            vertices = PatternDataset._generate_random_polygon(
                n_verts=n_verts,
                min_size=10.0,
                max_size=40.0,
                irregularity=params["irregularity"],
            )

            grain = random.choice(list(GrainDirection))
            if grain == GrainDirection.BIAS:
                allowed_rotations = [0.0, 90.0, 180.0, 270.0]
                tolerance = 10.0
            elif grain == GrainDirection.ANY:
                allowed_rotations = [0.0, 90.0, 180.0, 270.0]
                tolerance = 45.0
            else:
                allowed_rotations = [0.0, 180.0]
                tolerance = 5.0

            color = PIECE_COLORS[i % len(PIECE_COLORS)]

            piece = PatternPiece(
                id=f"random_{i+1:02d}",
                name=f"Random Piece {i+1}",
                vertices=vertices,
                grain_direction=grain,
                grain_tolerance_deg=tolerance,
                allowed_rotations=allowed_rotations,
                quantity=1,
                color=color,
            )
            pieces.append(piece)

        return pieces

    @staticmethod
    def _generate_random_polygon(
        n_verts: int,
        min_size: float,
        max_size: float,
        irregularity: float,
    ) -> List[tuple]:
        """Generate a random convex-ish polygon with given properties.

        Uses a radial point generation approach with controlled irregularity.

        Args:
            n_verts: Number of vertices.
            min_size: Minimum bounding dimension in cm.
            max_size: Maximum bounding dimension in cm.
            irregularity: How irregular the shape is (0=regular, 1=highly irregular).

        Returns:
            List of (x, y) vertex coordinates.
        """
        cx = random.uniform(min_size / 2, max_size / 2)
        cy = random.uniform(min_size / 2, max_size / 2)

        rx = random.uniform(min_size / 2, max_size / 2)
        ry = random.uniform(min_size / 2, max_size / 2)

        angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(n_verts)])

        vertices = []
        for angle in angles:
            # Add irregularity
            r_x = rx * (1 + random.uniform(-irregularity, irregularity))
            r_y = ry * (1 + random.uniform(-irregularity, irregularity))
            x = cx + r_x * math.cos(angle)
            y = cy + r_y * math.sin(angle)
            vertices.append((round(x, 2), round(y, 2)))

        # Normalize so bounding box starts at (0, 0)
        min_x = min(v[0] for v in vertices)
        min_y = min(v[1] for v in vertices)
        vertices = [(v[0] - min_x, v[1] - min_y) for v in vertices]

        return vertices

    @staticmethod
    def load_from_yaml(path: str) -> List[PatternPiece]:
        """Load a pattern piece dataset from a YAML file.

        Expected YAML format:
        ```yaml
        pieces:
          - id: "front_bodice"
            name: "Front Bodice"
            vertices: [[0, 0], [40, 0], [42, 30], [35, 60], [5, 60], [0, 30]]
            grain_direction: "vertical"
            grain_tolerance_deg: 5.0
            allowed_rotations: [0, 180]
            quantity: 1
            color: "#E74C3C"
        ```

        Args:
            path: Path to the YAML file.

        Returns:
            List of PatternPiece instances loaded from the file.

        Raises:
            ImportError: If PyYAML is not installed.
            FileNotFoundError: If the file does not exist.
        """
        if not HAS_YAML:
            raise ImportError("PyYAML is required for load_from_yaml. Install with: pip install pyyaml")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        pieces = []
        for piece_data in data.get("pieces", []):
            # Convert vertices from list of lists to list of tuples
            if "vertices" in piece_data:
                piece_data["vertices"] = [tuple(v) for v in piece_data["vertices"]]

            # Convert grain_direction string to enum
            if "grain_direction" in piece_data:
                piece_data["grain_direction"] = GrainDirection(piece_data["grain_direction"])

            piece = PatternPiece(**piece_data)
            pieces.append(piece)

        return pieces

    @staticmethod
    def get_total_piece_area(pieces: List[PatternPiece]) -> float:
        """Compute the total area of all pieces (accounting for quantity).

        Args:
            pieces: List of PatternPiece instances.

        Returns:
            Total area in cm².
        """
        return sum(p.area * p.quantity for p in pieces)

    @staticmethod
    def expand_pieces_by_quantity(pieces: List[PatternPiece]) -> List[PatternPiece]:
        """Expand pieces by their quantity, creating individual instances.

        For a piece with quantity=2, returns two separate pieces with suffixed IDs.

        Args:
            pieces: List of PatternPiece instances with quantity fields.

        Returns:
            List with each piece repeated according to its quantity.
        """
        expanded = []
        for piece in pieces:
            if piece.quantity == 1:
                expanded.append(piece)
            else:
                for q in range(piece.quantity):
                    new_piece = PatternPiece(
                        id=f"{piece.id}_copy{q+1}",
                        name=f"{piece.name} ({q+1}/{piece.quantity})",
                        vertices=piece.vertices,
                        grain_direction=piece.grain_direction,
                        grain_tolerance_deg=piece.grain_tolerance_deg,
                        allowed_rotations=piece.allowed_rotations,
                        quantity=1,
                        color=piece.color,
                    )
                    expanded.append(new_piece)
        return expanded
