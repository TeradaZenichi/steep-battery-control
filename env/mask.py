from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class StateMask:
    """Utility to build observation masks for SmartHomeEnv.

    The default layout (29 features) is documented below and can be
    customized by toggling each block.
    """

    include_climate: bool = True
    include_time: bool = True
    include_bess: bool = True
    include_ev: bool = True
    include_ev_shortfall: bool = True
    include_pv: bool = False
    include_load: bool = False
    include_tariff: bool = True
    include_net_load: bool = False
    include_steps_left: bool = True

    @staticmethod
    def layout() -> List[str]:
        return [
            "climate: drybulb",
            "climate: relhum",
            "climate: ghi",
            "climate: dni",
            "climate: dhi",
            "climate: wind",
            "time: month_sin",
            "time: month_cos",
            "time: day_sin",
            "time: day_cos",
            "time: hour_sin",
            "time: hour_cos",
            "time: minute_sin",
            "time: minute_cos",
            "time: weekday_sin",
            "time: weekday_cos",
            "bess: soc",
            "bess: soh",
            "ev: soc",
            "ev: soh",
            "ev: connected",
            "ev: shortfall",
            "pv: available_norm",
            "pv: used_norm",
            "pv: curtailment",
            "load_norm",
            "tariff_norm",
            "net_load_norm",
            "steps_left_ratio",
        ]

    def build(self) -> List[bool]:
        sections = []
        sections += [self.include_climate] * 6
        sections += [self.include_time] * 10
        sections += [self.include_bess] * 2
        sections += [self.include_ev] * 3
        sections += [self.include_ev_shortfall]
        sections += [self.include_pv] * 3
        sections += [self.include_load]
        sections += [self.include_tariff]
        sections += [self.include_net_load]
        sections += [self.include_steps_left]
        return sections



def get_state_mask() -> List[bool]:
    """Factory helper so callers can import a single symbol."""
    builder = StateMask()
    return builder.build()
