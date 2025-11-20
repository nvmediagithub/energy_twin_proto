from __future__ import annotations
from dataclasses import dataclass
from math import sin, pi

@dataclass(frozen=True)
class LoadProfiles:
    """
    Простые суточные профили 0..24.
    Выход: относительный множитель.
    """

    @staticmethod
    def multiplier(profile: str, hour: float) -> float:
        h = hour % 24.0

        if profile == "residential":
            # утро + вечер
            return 0.6 + 0.25*sin((h-7)/24*2*pi) + 0.35*sin((h-19)/24*2*pi)**2
        if profile == "commercial":
            # рабочий день 9-18
            if 8 <= h <= 19:
                return 1.2
            return 0.4
        if profile == "industrial":
            return 0.9  # почти ровно
        if profile == "nightlife":
            if 18 <= h or h <= 3:
                return 1.5
            return 0.5

        return 1.0
