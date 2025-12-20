from app.core.config import configs
from app.core.uvicorn_config import uvicorn_settings

if __name__ == "__main__":
    import uvicorn

    environment_settings = {"reload": True if configs.ENVIRONMENT != "production" else False}

    uvicorn.run(
        app="app:app",
        host=configs.APP_HOST,
        port=configs.APP_PORT,
        **uvicorn_settings,
        **environment_settings,
    )
