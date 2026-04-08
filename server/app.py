# Re-export the main FastAPI app from its package location.
# This file exists to satisfy the openenv validate check for server/app.py at repo root.
from incident_env.server.app import app  # noqa: F401
