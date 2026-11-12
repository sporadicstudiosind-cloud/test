"""Unit and dimension semantics for Iridium-1.

Implements the distinctions Architecture §5.1 requires:

* canonical base-dimension order ``(M, L, T, I, Theta, N, J)``
* rational exponents
* affine units (Celsius, Fahrenheit) with scale *and* offset
* three distinct states: non-numeric (``None``), dimensionless (all-zero
  exponents), and dimensional (some non-zero exponent)

The third point is a correction to the archived ORRERY specification, which
used a zero exponent vector for text as well as for dimensionless numbers.
Those are different things: ``None`` means "units do not apply", a zero vector
means "this is a number with no dimension".
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence

BASE_DIMENSIONS: tuple[str, ...] = (
    "mass",
    "length",
    "time",
    "current",
    "temperature",
    "amount",
    "luminous_intensity",
)

_SYMBOLS: tuple[str, ...] = ("M", "L", "T", "I", "Th", "N", "J")


class UnitError(ValueError):
    """Raised when a unit operation is not defined."""


@dataclass(frozen=True)
class Dimension:
    """Exponents over the seven SI base dimensions, in canonical order."""

    exponents: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        if len(self.exponents) != len(BASE_DIMENSIONS):
            raise UnitError(
                f"expected {len(BASE_DIMENSIONS)} exponents, got {len(self.exponents)}"
            )
        if not all(isinstance(e, Fraction) for e in self.exponents):
            raise UnitError("exponents must be Fraction instances")

    @classmethod
    def of(cls, **kwargs: float | int | str | Fraction) -> "Dimension":
        """Build from named base dimensions, e.g. ``Dimension.of(length=1, time=-1)``."""
        unknown = set(kwargs) - set(BASE_DIMENSIONS)
        if unknown:
            raise UnitError(f"unknown base dimensions: {sorted(unknown)}")
        return cls(
            tuple(Fraction(kwargs.get(name, 0)) for name in BASE_DIMENSIONS)
        )

    @classmethod
    def dimensionless(cls) -> "Dimension":
        return cls(tuple(Fraction(0) for _ in BASE_DIMENSIONS))

    @property
    def is_dimensionless(self) -> bool:
        return all(e == 0 for e in self.exponents)

    def __mul__(self, other: "Dimension") -> "Dimension":
        return Dimension(tuple(a + b for a, b in zip(self.exponents, other.exponents)))

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return Dimension(tuple(a - b for a, b in zip(self.exponents, other.exponents)))

    def __pow__(self, power: float | int | str | Fraction) -> "Dimension":
        p = Fraction(power)
        return Dimension(tuple(e * p for e in self.exponents))

    def __str__(self) -> str:
        if self.is_dimensionless:
            return "1"
        parts = []
        for sym, exp in zip(_SYMBOLS, self.exponents):
            if exp == 0:
                continue
            parts.append(sym if exp == 1 else f"{sym}^{exp}")
        return "·".join(parts)


# Common dimensions used across the physics and reporting contracts.
DIMENSIONLESS = Dimension.dimensionless()
LENGTH = Dimension.of(length=1)
TIME = Dimension.of(time=1)
MASS = Dimension.of(mass=1)
TEMPERATURE = Dimension.of(temperature=1)
VELOCITY = Dimension.of(length=1, time=-1)
ACCELERATION = Dimension.of(length=1, time=-2)
AREA = Dimension.of(length=2)
VOLUME = Dimension.of(length=3)
DENSITY = Dimension.of(mass=1, length=-3)
PRESSURE = Dimension.of(mass=1, length=-1, time=-2)
VOLUME_FLOW = Dimension.of(length=3, time=-1)
MASS_FLOW = Dimension.of(mass=1, time=-1)
MOMENTUM = Dimension.of(mass=1, length=1, time=-1)
ENERGY = Dimension.of(mass=1, length=2, time=-2)
POWER = Dimension.of(mass=1, length=2, time=-3)
DYNAMIC_VISCOSITY = Dimension.of(mass=1, length=-1, time=-1)
SURFACE_TENSION = Dimension.of(mass=1, time=-2)
HEAT_CAPACITY = Dimension.of(length=2, time=-2, temperature=-1)


@dataclass(frozen=True)
class Unit:
    """A named unit: ``value_in_SI = scale * value + offset``.

    ``offset`` supports affine units. Architecture §5.1 requires it explicitly,
    because ``degC`` is not a scaled kelvin.
    """

    name: str
    dimension: Dimension
    scale: float = 1.0
    offset: float = 0.0

    @property
    def is_affine(self) -> bool:
        return self.offset != 0.0

    def to_si(self, value: float) -> float:
        return self.scale * value + self.offset

    def from_si(self, value: float) -> float:
        return (value - self.offset) / self.scale


REGISTRY: dict[str, Unit] = {}


def register(unit: Unit) -> Unit:
    if unit.name in REGISTRY:
        raise UnitError(f"unit already registered: {unit.name}")
    REGISTRY[unit.name] = unit
    return unit


for _u in (
    Unit("1", DIMENSIONLESS),
    Unit("m", LENGTH),
    Unit("mm", LENGTH, 1e-3),
    Unit("km", LENGTH, 1e3),
    Unit("s", TIME),
    Unit("ms", TIME, 1e-3),
    Unit("kg", MASS),
    Unit("g", MASS, 1e-3),
    Unit("K", TEMPERATURE),
    Unit("degC", TEMPERATURE, 1.0, 273.15),
    Unit("degF", TEMPERATURE, 5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0),
    Unit("m/s", VELOCITY),
    Unit("m2", AREA),
    Unit("m3", VOLUME),
    Unit("kg/m3", DENSITY),
    Unit("Pa", PRESSURE),
    Unit("m3/s", VOLUME_FLOW),
    Unit("kg/s", MASS_FLOW),
    Unit("J", ENERGY),
    Unit("W", POWER),
    Unit("Pa.s", DYNAMIC_VISCOSITY),
    Unit("N/m", SURFACE_TENSION),
    Unit("J/(kg.K)", HEAT_CAPACITY),
):
    register(_u)


@dataclass(frozen=True)
class Quantity:
    """A numeric value with a unit, or a non-numeric value with ``unit=None``."""

    value: float
    unit: Unit | None

    @property
    def is_numeric(self) -> bool:
        return self.unit is not None

    @property
    def is_dimensionless(self) -> bool:
        """True only for numbers with a zero exponent vector.

        Non-numeric data (``unit is None``) is *not* dimensionless; it has no
        unit semantics at all.
        """
        return self.unit is not None and self.unit.dimension.is_dimensionless

    def to_si(self) -> float:
        if self.unit is None:
            raise UnitError("non-numeric quantity has no SI value")
        return self.unit.to_si(self.value)

    def to(self, target: Unit | str) -> "Quantity":
        target_unit = REGISTRY[target] if isinstance(target, str) else target
        if self.unit is None:
            raise UnitError("cannot convert a non-numeric quantity")
        if self.unit.dimension != target_unit.dimension:
            raise UnitError(
                f"incompatible dimensions: {self.unit.dimension} -> {target_unit.dimension}"
            )
        return Quantity(target_unit.from_si(self.to_si()), target_unit)


def parse_dimension(spec: Mapping[str, float | int | str] | Sequence[float] | None):
    """Parse a dimension from an event payload.

    ``None`` maps to ``None`` (non-numeric). A mapping or a 7-sequence maps to a
    ``Dimension``. An empty mapping is dimensionless, not non-numeric.
    """
    if spec is None:
        return None
    if isinstance(spec, Mapping):
        return Dimension.of(**{k: Fraction(str(v)) for k, v in spec.items()})
    exponents = list(spec)
    if len(exponents) != len(BASE_DIMENSIONS):
        raise UnitError(
            f"dimension sequence must have {len(BASE_DIMENSIONS)} entries"
        )
    return Dimension(tuple(Fraction(str(e)) for e in exponents))
