import multiprocessing

from app.core.config import configs

is_prod = configs.ENVIRONMENT == "production"

uvicorn_settings = {
    "workers": min(multiprocessing.cpu_count(), 2) if is_prod else 1,
    "backlog": 4096,
    "timeout_keep_alive": 120,
}

if is_prod:
    uvicorn_settings.update({
        "loop": "uvloop",
        "http": "httptools",
        "access_log": True,
        "log_level": "info",
    })


# SSL 설정 (옵션)
# ssl_certfile = None  # SSL 인증서 경로
# ssl_keyfile = None   # SSL 키 경로
# if ssl_certfile and ssl_keyfile:
#     uvicorn_settings["ssl_certfile"] = ssl_certfile
#     uvicorn_settings["ssl_keyfile"] = ssl_keyfile
