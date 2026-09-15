from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

PRESSURE_VARIABLES = [
    "geopotential",
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
]

SURFACE_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_precipitation_6hr",
]


@dataclass(frozen=True)
class ChannelLayout:
    levels_hpa: tuple[float, ...]
    names: tuple[str, ...]
    variable_slices: Dict[str, slice]

    @property
    def n_channels(self) -> int:
        return len(self.names)

    def indices(self, variable: str) -> list[int]:
        sl = self.variable_slices[variable]
        return list(range(sl.start, sl.stop))

    def one(self, variable: str) -> int:
        sl = self.variable_slices[variable]
        if sl.stop - sl.start != 1:
            raise ValueError(f"{variable} has multiple channels")
        return sl.start

    def to_dict(self) -> dict:
        return {
            "levels_hpa": list(self.levels_hpa),
            "names": list(self.names),
            "variable_slices": {
                k: [v.start, v.stop]
                for k, v in self.variable_slices.items()
            },
        }


def build_layout(levels_hpa: Sequence[float]) -> ChannelLayout:
    names: List[str] = []
    slices: Dict[str, slice] = {}
    cursor = 0

    for variable in PRESSURE_VARIABLES:
        start = cursor
        for level in levels_hpa:
            names.append(f"{variable}@{float(level):g}hPa")
            cursor += 1
        slices[variable] = slice(start, cursor)

    for variable in SURFACE_VARIABLES:
        start = cursor
        names.append(variable)
        cursor += 1
        slices[variable] = slice(start, cursor)

    return ChannelLayout(
        levels_hpa=tuple(float(x) for x in levels_hpa),
        names=tuple(names),
        variable_slices=slices,
    )
