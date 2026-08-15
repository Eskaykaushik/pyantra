"""JSON Schema mapping for MCP tool manifests."""

from __future__ import annotations

from typing import Any

_TYPE_MAP: dict[str, type[Any]] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def json_schema_to_python(schema: dict[str, Any]) -> dict[str, type[Any]]:
    """Map a tool's ``inputSchema`` (a JSON Schema object) to Python types.

    Returns ``{property_name: python_type}`` for every entry in the schema's
    ``properties``. Properties without a resolvable ``type`` map to ``Any``.
    Union type arrays map to ``Any`` when the members disagree.
    """
    result: dict[str, type[Any]] = {}
    for name, prop in (schema.get("properties") or {}).items():
        result[name] = _json_type_to_python(prop)
    return result


def required_properties(schema: dict[str, Any]) -> set[str]:
    """Return the set of property names the schema marks as required."""
    return set(schema.get("required") or [])


def _json_type_to_python(prop: dict[str, Any]) -> type[Any]:
    raw = prop.get("type")
    if isinstance(raw, list):
        types = [_TYPE_MAP[t] for t in raw if isinstance(t, str) and t in _TYPE_MAP]
        non_null = [t for t in types if t is not type(None)]
        if len(non_null) == 1:
            return non_null[0]
        if not types:
            return Any
        return Any
    if isinstance(raw, str) and raw in _TYPE_MAP:
        return _TYPE_MAP[raw]
    return Any


__all__ = ["json_schema_to_python", "required_properties"]
