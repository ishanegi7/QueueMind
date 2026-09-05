"""API route handlers package for QueueMind."""

from api.routes.congestion import router as congestion_router
from api.routes.health import router as health_router
from api.routes.patient_flow import router as patient_flow_router
from api.routes.queue_health import router as queue_health_router
from api.routes.simulation import router as simulation_router

__all__ = [
    "health_router",
    "patient_flow_router",
    "congestion_router",
    "queue_health_router",
    "simulation_router",
]
