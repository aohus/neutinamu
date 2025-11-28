import json
import logging
import logging.config
import sys

from app.core.config import settings


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


def setup_logging():
    """
    Set up logging configuration based on the environment.
    """
    log_level = settings.LOG_LEVEL.upper()
    
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {
                "()": JsonFormatter,
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "default",
            },
            "json": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "json",
            },
        },
        "loggers": {
            "app": {
                "level": log_level,
                "handlers": ["default"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["default"],
                "propagate": False,
            },
            "uvicorn.error": {
                "level": "ERROR",
                "handlers": ["default"],
                "propagate": False,
            },
            "sqlalchemy.engine": {
                "level": "WARNING",
                "handlers": ["default"],
                "propagate": False,
            }
        },
        "root": {
            "level": log_level,
            "handlers": ["default"]
        },
    }

    if settings.ENVIRONMENT == "production":
        logging_config["loggers"]["app"]["handlers"] = ["json"]
        logging_config["loggers"]["uvicorn.access"]["handlers"] = ["json"]
        logging_config["loggers"]["uvicorn.error"]["handlers"] = ["json"]
        logging_config["root"]["handlers"] = ["json"]


    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete for {settings.ENVIRONMENT} environment with level {log_level}")