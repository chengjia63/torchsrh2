from collections import namedtuple
from typing import List, Dict
import numpy as np
import itertools


class TailwindColors:
    """Helper for sampling colors from a fixed set of Tailwind-like ramps.

    Common usage:
    - ``TC()(c="RALTFSV")``: choose explicit color families by symbol.
    - ``TC()(c="RALTFSV", s=6)``: choose explicit families at one shared shade.
    - ``TC()(c="RALTFSV", s=[6, 6, 6, 6, 6, 6, 6])``: choose explicit shades.
    - ``TC()(nc=7)``: sample ``7`` distinct families from the default order.

    Shade indices run from ``0`` (lightest) to ``10`` (darkest).
    Keep ``color_map`` and the ``color_symbols`` order in ``__call__`` aligned.
    """

    white: str = "#ffffff"
    black: str = "#000000"

    slate: List[str] = [
        "#f8fafc",
        "#f1f5f9",
        "#e2e8f0",
        "#cbd5e1",
        "#94a3b8",
        "#64748b",
        "#475569",
        "#334155",
        "#1e293b",
        "#0f172a",
        "#020617",
    ]

    gray: List[str] = [
        "#f9fafb",
        "#f3f4f6",
        "#e5e7eb",
        "#d1d5db",
        "#9ca3af",
        "#6b7280",
        "#4b5563",
        "#374151",
        "#1f2937",
        "#111827",
        "#030712",
    ]

    zinc: List[str] = [
        "#fafafa",
        "#f4f4f5",
        "#e4e4e7",
        "#d4d4d8",
        "#a1a1aa",
        "#71717a",
        "#52525b",
        "#3f3f46",
        "#27272a",
        "#18181b",
        "#09090b",
    ]

    neutral: List[str] = [
        "#fafafa",
        "#f5f5f5",
        "#e5e5e5",
        "#d4d4d4",
        "#a3a3a3",
        "#737373",
        "#525252",
        "#404040",
        "#262626",
        "#171717",
        "#0a0a0a",
    ]

    stone: List[str] = [
        "#fafaf9",
        "#f5f5f4",
        "#e7e5e4",
        "#d6d3d1",
        "#a8a29e",
        "#78716c",
        "#57534e",
        "#44403c",
        "#292524",
        "#1c1917",
        "#0c0a09",
    ]

    red: List[str] = [
        "#fef2f2",
        "#fee2e2",
        "#fecaca",
        "#fca5a5",
        "#f87171",
        "#ef4444",
        "#dc2626",
        "#b91c1c",
        "#991b1b",
        "#7f1d1d",
        "#450a0a",
    ]

    orange: List[str] = [
        "#fff7ed",
        "#ffedd5",
        "#fed7aa",
        "#fdba74",
        "#fb923c",
        "#f97316",
        "#ea580c",
        "#c2410c",
        "#9a3412",
        "#7c2d12",
        "#431407",
    ]

    amber: List[str] = [
        "#fffbeb",
        "#fef3c7",
        "#fde68a",
        "#fcd34d",
        "#fbbf24",
        "#f59e0b",
        "#d97706",
        "#b45309",
        "#92400e",
        "#78350f",
        "#451a03",
    ]

    yellow: List[str] = [
        "#fefce8",
        "#fef9c3",
        "#fef08a",
        "#fde047",
        "#facc15",
        "#eab308",
        "#ca8a04",
        "#a16207",
        "#854d0e",
        "#713f12",
        "#422006",
    ]

    lime: List[str] = [
        "#f7fee7",
        "#ecfccb",
        "#d9f99d",
        "#bef264",
        "#a3e635",
        "#84cc16",
        "#65a30d",
        "#4d7c0f",
        "#3f6212",
        "#365314",
        "#1a2e05",
    ]

    green: List[str] = [
        "#f0fdf4",
        "#dcfce7",
        "#bbf7d0",
        "#86efac",
        "#4ade80",
        "#22c55e",
        "#16a34a",
        "#15803d",
        "#166534",
        "#14532d",
        "#052e16",
    ]

    emerald: List[str] = [
        "#ecfdf5",
        "#d1fae5",
        "#a7f3d0",
        "#6ee7b7",
        "#34d399",
        "#10b981",
        "#059669",
        "#047857",
        "#065f46",
        "#064e3b",
        "#022c22",
    ]

    teal: List[str] = [
        "#f0fdfa",
        "#ccfbf1",
        "#99f6e4",
        "#5eead4",
        "#2dd4bf",
        "#14b8a6",
        "#0d9488",
        "#0f766e",
        "#115e59",
        "#134e4a",
        "#042f2e",
    ]

    cyan: List[str] = [
        "#ecfeff",
        "#cffafe",
        "#a5f3fc",
        "#67e8f9",
        "#22d3ee",
        "#06b6d4",
        "#0891b2",
        "#0e7490",
        "#155e75",
        "#164e63",
        "#083344",
    ]

    sky: List[str] = [
        "#f0f9ff",
        "#e0f2fe",
        "#bae6fd",
        "#7dd3fc",
        "#38bdf8",
        "#0ea5e9",
        "#0284c7",
        "#0369a1",
        "#075985",
        "#0c4a6e",
        "#082f49",
    ]

    blue: List[str] = [
        "#eff6ff",
        "#dbeafe",
        "#bfdbfe",
        "#93c5fd",
        "#60a5fa",
        "#3b82f6",
        "#2563eb",
        "#1d4ed8",
        "#1e40af",
        "#1e3a8a",
        "#172554",
    ]

    indigo: List[str] = [
        "#eef2ff",
        "#e0e7ff",
        "#c7d2fe",
        "#a5b4fc",
        "#818cf8",
        "#6366f1",
        "#4f46e5",
        "#4338ca",
        "#3730a3",
        "#312e81",
        "#1e1b4b",
    ]

    violet: List[str] = [
        "#f5f3ff",
        "#ede9fe",
        "#ddd6fe",
        "#c4b5fd",
        "#a78bfa",
        "#8b5cf6",
        "#7c3aed",
        "#6d28d9",
        "#5b21b6",
        "#4c1d95",
        "#2e1065",
    ]

    purple: List[str] = [
        "#faf5ff",
        "#f3e8ff",
        "#e9d5ff",
        "#d8b4fe",
        "#c084fc",
        "#a855f7",
        "#9333ea",
        "#7e22ce",
        "#6b21a8",
        "#581c87",
        "#3b0764",
    ]

    fuchsia: List[str] = [
        "#fdf4ff",
        "#fae8ff",
        "#f5d0fe",
        "#f0abfc",
        "#e879f9",
        "#d946ef",
        "#c026d3",
        "#a21caf",
        "#86198f",
        "#701a75",
        "#4a044e",
    ]

    pink: List[str] = [
        "#fdf2f8",
        "#fce7f3",
        "#fbcfe8",
        "#f9a8d4",
        "#f472b6",
        "#ec4899",
        "#db2777",
        "#be185d",
        "#9d174d",
        "#831843",
        "#500724",
    ]

    rose: List[str] = [
        "#fff1f2",
        "#ffe4e6",
        "#fecdd3",
        "#fda4af",
        "#fb7185",
        "#f43f5e",
        "#e11d48",
        "#be123c",
        "#9f1239",
        "#881337",
        "#4c0519",
    ]

    color_map = {
        "R": red,
        "O": orange,
        "A": amber,
        "Y": yellow,
        "L": lime,
        "G": green,
        "E": emerald,
        "T": teal,
        "C": cyan,
        "S": sky,
        "B": blue,
        "I": indigo,
        "V": violet,
        "P": purple,
        "F": fuchsia,
        "Q": pink,
        "W": rose,
        "M": slate,
        "H": gray,
        "Z": zinc,
        "N": neutral,
        "U": stone,
    }

    @staticmethod
    def retrieve_colors(colors, steps):
        """Resolve color-family symbols plus shade indices to hex strings."""
        assert len(colors) == len(steps)
        return [TailwindColors.color_map[c][s] for c, s in zip(colors, steps)]

    @staticmethod
    def get_steps(
        n: int,
        s: int | list[int] | None = 6,
    ) -> list[int]:
        if isinstance(s, int):
            return [s] * n
        return [int(v) for v in s]

    @staticmethod
    def __call__(
        c: str | None = None,
        nc: int | None = None,
        s: int | list[int] | None = 6,
    ) -> List[str]:
        """Return colors by explicit symbols or by sampled family count."""

        if c:
            return TailwindColors.retrieve_colors(
                c, TailwindColors.get_steps(len(c), s)
            )

        elif nc:
            # Default family order used when sampling ``nc`` colors.
            color_symbols = "ROAYLGETCSBIVPFQW"
            max_base = len(color_symbols)

            if nc <= max_base:
                s = TailwindColors.get_steps(nc, s)

                indices = np.linspace(
                    0, len(color_symbols), nc, endpoint=False, dtype=int
                )
                sampled_colors = [color_symbols[i] for i in indices]
                return TailwindColors.retrieve_colors(sampled_colors, s)

            else:
                raise NotImplementedError()

        else:
            raise ValueError("Must provide either `colors` or `nc`.")


TC = TailwindColors
