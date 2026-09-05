"""Main FastAPI application factory and entrypoint for QueueMind API."""

from __future__ import annotations

import logging
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import Settings, get_settings
from api.routes import (
    congestion_router,
    health_router,
    patient_flow_router,
    queue_health_router,
    simulation_router,
)

logger = logging.getLogger("queuemind.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the QueueMind FastAPI application instance."""
    app_settings = settings or get_settings()

    # Configure root logging
    logging.basicConfig(
        level=getattr(logging, app_settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app = FastAPI(
        title="QueueMind API",
        description=(
            "AI-Powered Emergency Department Patient Flow Intelligence API. "
            "Provides patient journey forecasting, multi-horizon congestion "
            "forecasts, TreeSHAP explainability, Queue Health scoring, "
            "and operational what-if simulations."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    if settings is not None:
        app.dependency_overrides[get_settings] = lambda: app_settings

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global Exception Handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle request schema validation errors cleanly without leaking internals."""
        logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation Error",
                "detail": jsonable_encoder(exc.errors()),
                "path": request.url.path,
            },
        )

    @app.exception_handler(Exception)
    async def uncaught_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch unhandled exceptions and prevent leaking stack traces or secrets."""
        logger.error(
            "Unhandled server error on %s: %s", request.url.path, exc, exc_info=True
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "message": (
                    "An unexpected error occurred during request processing. "
                    "The incident has been logged."
                ),
                "path": request.url.path,
            },
        )

    # Include API Routers
    app.include_router(health_router)
    app.include_router(patient_flow_router)
    app.include_router(congestion_router)
    app.include_router(queue_health_router)
    app.include_router(simulation_router)

    return app


# Default application instance for Uvicorn
app = create_app()
