"""FastAPI entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from industrial_safety_vision.api.routes import InferenceService, router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Industrial Safety Vision API",
        description="Real-time industrial safety monitoring inference service.",
        version="0.1.0",
    )
    app.state.service = InferenceService()
    app.include_router(router)
    return app


app = create_app()
