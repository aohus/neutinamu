# Logging Configuration

This document explains how to configure the logging for the application based on the environment.

## Overview

The application uses Python's built-in `logging` module. The logging configuration is determined by environment variables at startup. This allows for flexible logging depending on whether you are running in a development, testing, or production environment.

## Configuration

The logging behavior is controlled by two main environment variables:

- `ENVIRONMENT`: Determines the overall logging format.
  - `development` (default): Human-readable, colored console logs. Best for local development.
  - `production`: JSON-formatted logs sent to `stdout`. Best for containerized environments where logs are collected and processed by an external service (e.g., Docker, Datadog, ELK stack).
  - `testing`: Same as development but can be configured for less verbose output.

- `LOG_LEVEL`: Determines the minimum level of logs to be emitted.
  - `DEBUG`: Detailed information, typically of interest only when diagnosing problems.
  - `INFO` (default): Confirmation that things are working as expected.
  - `WARNING`: An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected.
  - `ERROR`: Due to a more serious problem, the software has not been able to perform some function.
  - `CRITICAL`: A serious error, indicating that the program itself may be unable to continue running.

### How to Set Environment Variables

You can set these variables in a `.env` file in the project root directory, or directly in your `docker-compose.yml` file.

#### Using `.env` file

Create or edit the `.env` file in the `/backend` directory:

```
# .env

# Set environment to 'development' or 'production'
ENVIRONMENT=development

# Set the desired log level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=DEBUG

# Other application settings...
SECRET_KEY=your-secret-key
ALGORITHM=HS256
```

#### Using `docker-compose.yml`

You can also set these variables directly in the `docker-compose.yml` file for the `backend` service:

```yaml
services:
  backend:
    build:
      context: ./src
      dockerfile: Dockerfile
    environment:
      # ... other environment variables
      ENVIRONMENT: production
      LOG_LEVEL: INFO
    # ... rest of the service definition
```

Variables set in `docker-compose.yml` will override those in the `.env` file.

## Log Formats

### Development Example

When `ENVIRONMENT=development`, logs are easy to read in the console.

```
[2023-10-27 10:30:00] [INFO] [app.main] Starting application lifespan...
[2023-10-27 10:30:00] [INFO] [app.main] Executors initialized and data directory ensured.
[2023-10-27 10:30:01] [INFO] [uvicorn.error] Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
[2023-10-27 10:30:05] [INFO] [app.api.endpoints.auth] Login attempt for username: testuser
[2023-10-27 10:30:05] [DEBUG] [app.core.security] Verifying password.
[2023-10-27 10:30:05] [DEBUG] [app.core.security] Creating new access token for subject: 1234-5678-9101
[2023-10-27 10:30:05] [INFO] [app.api.endpoints.auth] Login successful for user: testuser
```

### Production Example

When `ENVIRONMENT=production`, logs are formatted as JSON, which is ideal for log aggregation and analysis tools.

```json
{"timestamp": "2023-10-27T10:30:00+0000", "level": "INFO", "name": "app.main", "message": "Starting application lifespan..."}
{"timestamp": "2023-10-27T10:30:00+0000", "level": "INFO", "name": "app.main", "message": "Executors initialized and data directory ensured."}
{"timestamp": "2023-10-27T10:30:01+0000", "level": "INFO", "name": "uvicorn.error", "message": "Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)"}
{"timestamp": "2023-10-27T10:30:05+0000", "level": "INFO", "name": "app.api.endpoints.auth", "message": "Login attempt for username: testuser"}
{"timestamp": "2023-10-27T10:30:05+0000", "level": "INFO", "name": "app.api.endpoints.auth", "message": "Login successful for user: testuser"}
```

This structured format allows you to easily filter, search, and visualize logs based on fields like `level`, `name`, and `timestamp`.
