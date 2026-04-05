"""
server/app.py — OpenEnv-compatible entry point for the Student Task Manager.

This module is used by openenv validate and openenv serve.
The main() function starts the FastAPI server via uvicorn.

All routes are registered directly here (imported from the root server.py logic)
or we start uvicorn pointing at the root server module.
"""

from __future__ import annotations

import importlib.util
import os
import sys

# Ensure project root is on path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

# Load main.py by file path to avoid any package naming conflicts
_server_spec = importlib.util.spec_from_file_location(
    "_root_main", os.path.join(_root, "main.py")
)
_server_mod = importlib.util.module_from_spec(_server_spec)  # type: ignore[arg-type]
_server_spec.loader.exec_module(_server_mod)  # type: ignore[union-attr]
app = _server_mod.app  # FastAPI application instance


def main() -> None:
    """
    Entry point for openenv serve / uv run / direct execution.

    Starts uvicorn with the FastAPI app on the configured host and port.
    Override via environment variables:
      HOST  (default: 0.0.0.0)
      PORT  (default: 8000)
    """
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
