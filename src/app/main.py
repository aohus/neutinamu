from app.core.config import configs
from app.core.uvicorn_config import uvicorn_settings
from app.core.logger import setup_logging
from app import create_app

# Setup logging before creating the app
setup_logging()

# Create the application instance
app = create_app()

if __name__ == "__main__":
    import uvicorn

    environment_settings = {"reload": True if configs.ENVIRONMENT != "production" else False}

    uvicorn.run(
        app="app.main:app",
        host=configs.APP_HOST,
        port=configs.APP_PORT,
        **uvicorn_settings,
        **environment_settings,
    )
