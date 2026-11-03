"""The canonical error contract, the Result type, and named limits.

Every failure the model or a user sees is a ``HarnessError``: a code, a
category, a message, whether it is retryable, the remediation action, an
optional bounded recovery step, details, and whatever partial result
survived. The rule to the model is always: do not repeat the unchanged call.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E")


class ErrorCategory(str, Enum):  # noqa: UP042  (str mixin keeps JSON and comparisons simple)
    invalid_input = "invalid_input"
    permission_denied = "permission_denied"
    not_found = "not_found"
    unsupported = "unsupported"
    truncated = "truncated"
    cancelled = "cancelled"
    timeout = "timeout"
    resource_constraint = "resource_constraint"
    unavailable = "unavailable"
    execution_failure = "execution_failure"
    integrity = "integrity"
    configuration_invalid = "configuration_invalid"


@dataclass
class HarnessError:
    code: str
    message: str
    category: ErrorCategory = ErrorCategory.execution_failure
    retryable: bool = False
    remediation: str = ""
    recovery: dict[str, Any] | None = None  # {"kind": "inspect_input_contract", "operation": ...}
    details: dict[str, Any] = field(default_factory=dict)
    partial: Any = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "code": self.code,
            "category": self.category.value if isinstance(self.category, ErrorCategory) else str(self.category),
            "message": self.message,
            "retryable": self.retryable,
            "remediation": self.remediation or "Do not repeat the unchanged call.",
        }
        if self.recovery:
            d["recovery"] = self.recovery
        if self.details:
            d["details"] = self.details
        if self.partial is not None:
            d["partial"] = self.partial
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HarnessError:
        cat = d.get("category", "execution_failure")
        try:
            category = ErrorCategory(cat)
        except ValueError:
            category = ErrorCategory.execution_failure
        return cls(
            code=d.get("code", "error"),
            message=d.get("message", ""),
            category=category,
            retryable=bool(d.get("retryable", False)),
            remediation=d.get("remediation", ""),
            recovery=d.get("recovery"),
            details=dict(d.get("details") or {}),
            partial=d.get("partial"),
        )

    def __str__(self) -> str:
        s = f"{self.code}: {self.message}"
        if self.remediation:
            s += f" Remediation: {self.remediation}"
        return s


class ToolFailure(Exception):
    """Raise inside a tool to return a typed error instead of a stack trace."""

    def __init__(
        self,
        code: str,
        message: str,
        category: ErrorCategory | str = ErrorCategory.execution_failure,
        retryable: bool = False,
        remediation: str = "",
        recovery: dict[str, Any] | None = None,
        partial: Any = None,
        **details: Any,
    ) -> None:
        super().__init__(message)
        if isinstance(category, str):
            category = ErrorCategory(category)
        self.error = HarnessError(code, message, category, retryable, remediation, recovery, details, partial)

    @property
    def code(self) -> str:
        return self.error.code


class ConfigurationError(Exception):
    """An invalid setting. Never silently replaced by a default."""


# ----------------------------------------------------------------------------- Result
@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T
    ok: bool = True

    def unwrap(self) -> T:
        return self.value


@dataclass(frozen=True)
class Err(Generic[E]):
    error: E
    ok: bool = False

    def unwrap(self) -> Any:
        raise RuntimeError(f"unwrap on Err: {self.error}")


Result = Ok[T] | Err[E]


def ok(value: T) -> Ok[T]:
    return Ok(value)


def err(error: E) -> Err[E]:
    return Err(error)


# ----------------------------------------------------------------------------- Limits
@dataclass(frozen=True)
class Limit:
    """A named ceiling with a unit and honest recovery text.

    Every budget, timeout and size ceiling in Edgent is one of these, so a failure
    can name the limit, its effective value and source, the next action, whether a
    restart is needed, and what does not raise it.
    """

    name: str
    value: float
    unit: str
    source: str = "default"  # default | setting
    setting_name: str | None = None
    floor: float | None = None
    maximum: float | None = None
    restart_required: bool = False
    does_not_raise: tuple[str, ...] = ()

    @classmethod
    def from_env(
        cls,
        name: str,
        default: float,
        unit: str,
        env: str | None = None,
        floor: float | None = None,
        maximum: float | None = None,
        restart_required: bool = False,
        does_not_raise: tuple[str, ...] = (),
        environ: dict[str, str] | None = None,
    ) -> Limit:
        environ = os.environ if environ is None else environ
        raw = environ.get(env) if env else None
        if raw is None or raw == "":
            return cls(name, default, unit, "default", env, floor, maximum, restart_required, does_not_raise)
        try:
            val = float(raw) if "." in raw else int(raw)
        except ValueError as e:
            raise ConfigurationError(f"{env} must be a number of {unit}; got {raw!r}. Correct it in the environment that starts Edgent.") from e
        if floor is not None and val < floor:
            raise ConfigurationError(f"{env}={val} is below the floor {floor} {unit}. Correct it in the environment that starts Edgent.")
        if maximum is not None and val > maximum:
            raise ConfigurationError(f"{env}={val} is above the maximum {maximum} {unit}. Correct it in the environment that starts Edgent.")
        return cls(name, val, unit, "setting", env, floor, maximum, restart_required, does_not_raise)

    def describe(self) -> str:
        s = f"The effective {self.name} is {self.value:g} {self.unit} (source {self.source}"
        if self.setting_name:
            s += f"; setting {self.setting_name}"
        if self.floor is not None:
            s += f"; floor {self.floor:g}"
        if self.maximum is not None:
            s += f"; maximum {self.maximum:g}"
        s += ")."
        return s

    def exceeded(self, observed: float, action: str, category: ErrorCategory = ErrorCategory.resource_constraint, **details: Any) -> HarnessError:
        msg = f"{self.name} exceeded: observed {observed:g} {self.unit}. {self.describe()}"
        remediation = action
        if self.setting_name:
            remediation += f" To raise it, set {self.setting_name} in the environment that starts Edgent"
            remediation += " and restart." if self.restart_required else "."
        else:
            remediation += " This release has no runtime override for this limit."
        if self.does_not_raise:
            remediation += " Does not raise it: " + ", ".join(self.does_not_raise) + "."
        return HarnessError(
            code=f"limit:{self.name}",
            message=msg,
            category=category,
            retryable=False,
            remediation=remediation,
            details={"limit": self.name, "value": self.value, "unit": self.unit, "observed": observed, **details},
        )
