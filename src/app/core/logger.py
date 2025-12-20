import json
import logging
import logging.config
import sys

from app.core.config import configs


class JsonFormatter(logging.Formatter):
    """
    Formatter for logging in JSON format.
    """

    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


# -----------------------------------------------------------------------------
# Development Logging Configuration
# -----------------------------------------------------------------------------
# Console-friendly, readable text format.
DEV_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "default",
        },
    },
    "loggers": {
        # Root Logger: Catches everything not caught by specific loggers
        "root": {
            "level": configs.LOG_LEVEL,
            "handlers": ["console"],
        },
        # Application Logger
        "app": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
        # Uvicorn (FastAPI Server) Loggers
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
        "uvicorn.error": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
        # SQLAlchemy (Database) Loggers
        # Set to INFO to see SQL queries, DEBUG for results
        "sqlalchemy.engine": {
            "level": "WARNING",  # Change to INFO or DEBUG for more verbosity
            "handlers": ["console"],
            "propagate": False,
        },
        "alembic": {
            "level": "ERROR",
            "handlers": ["console"],
            "propagate": False,
        },
    },
}

# -----------------------------------------------------------------------------
# Production Logging Configuration
# -----------------------------------------------------------------------------
# JSON structured, machine-parsable, suitable for aggregation (ELK, CloudWatch, etc.)
PROD_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": JsonFormatter,
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        },
    },
    "handlers": {
        "console_json": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "json",
        },
    },
    "loggers": {
        "root": {
            "level": configs.LOG_LEVEL,
            "handlers": ["console_json"],
        },
        "app": {
            "level": configs.LOG_LEVEL,
            "handlers": ["console_json"],
            "propagate": False,
        },
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console_json"],
            "propagate": False,
        },
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["console_json"],
            "propagate": False,
        },
        "uvicorn.error": {
            "level": "ERROR",
            "handlers": ["console_json"],
            "propagate": False,
        },
        "sqlalchemy.engine": {
            "level": "WARNING",
            "handlers": ["console_json"],
            "propagate": False,
        },
        "alembic": {
            "level": "WARNING",
            "handlers": ["console_json"],
            "propagate": False,
        },
    },
}


def setup_logging():
    """
    Set up logging configuration based on the environment.
    """
    env = configs.ENVIRONMENT.lower()

    if env == "production":
        log_config = PROD_LOGGING_CONFIG
    else:
        log_config = DEV_LOGGING_CONFIG

    # Apply configuration
    logging.config.dictConfig(log_config)

    # Log the setup confirmation (using a logger defined in the config)
    logger = logging.getLogger("app")
    logger.info(f"Logging setup complete for {env} environment with level {configs.LOG_LEVEL}")
