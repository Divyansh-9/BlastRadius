"""
server/app.py – OpenEnv entry point.

The openenv validate tool requires:
  - A main() function at this path
  - if __name__ == '__main__' block
  - [project.scripts] server = "server.app:main"
"""

import uvicorn
from incident_env.server.app import app  # noqa: F401


def main():
    """Start the FastAPI / uvicorn server on port 7860."""
    uvicorn.run(
        "incident_env.server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()
