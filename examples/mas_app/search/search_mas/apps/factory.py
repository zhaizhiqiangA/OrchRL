from __future__ import annotations

from typing import Any

from .base import BaseMASApplication


def build_application(config: dict[str, Any]) -> BaseMASApplication:
    app_type = str(config.get("application", {}).get("type", "search")).lower()
    if app_type == "search":
        from .search.app import SearchMASApplication

        return SearchMASApplication.from_config(config)
    raise ValueError(f"Unsupported application type: {app_type}")
