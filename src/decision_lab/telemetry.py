from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """Create an OpenTelemetry span when opentelemetry-api is installed.

    The core package remains dependency-light; without OpenTelemetry this becomes
    a no-op context manager.
    """
    try:
        from opentelemetry import trace
    except ImportError:
        yield None
        return

    tracer = trace.get_tracer("agent-decision-lab")
    with tracer.start_as_current_span(name) as current:
        for key, value in (attributes or {}).items():
            if value is not None:
                current.set_attribute(key, value)
        yield current


def set_attribute(current: Any, key: str, value: Any) -> None:
    if current is not None and value is not None:
        current.set_attribute(key, value)
