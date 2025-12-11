"""
Конфигурация Gunicorn для FastAPI приложения
"""

import multiprocessing
import os

# Количество воркеров
workers_per_core_str = os.getenv("WORKERS_PER_CORE", "1")
max_workers_str = os.getenv("MAX_WORKERS")
use_max_workers = None

if max_workers_str:
    use_max_workers = int(max_workers_str)

cores = multiprocessing.cpu_count()
workers_per_core = float(workers_per_core_str)
default_web_concurrency = workers_per_core * cores

if use_max_workers:
    web_concurrency = max(int(default_web_concurrency), use_max_workers)
else:
    web_concurrency = max(int(default_web_concurrency), 2)

# Gunicorn конфигурация
worker_class = "uvicorn.workers.UvicornWorker"
workers = web_concurrency
worker_connections = 1000

# Socket
bind = os.getenv("BIND", "0.0.0.0:8000")

# Логирование
accesslog = os.getenv("ACCESS_LOG", "-")
errorlog = os.getenv("ERROR_LOG", "-")
loglevel = os.getenv("LOG_LEVEL", "info")
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(L)s'

# Таймауты
keepalive = 120
timeout = 120
graceful_timeout = 120

# Перезагрузка воркеров
max_requests = 1000
max_requests_jitter = 100

# Безопасность
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# Отладка
reload = os.getenv("RELOAD", "false").lower() == "true"
spew = False
check_config = False

# Preload для ускорения запуска
preload_app = True

# PID файл
pidfile = "/tmp/gunicorn.pid"

# Daemon режим
daemon = False

# Пользователь
user = os.getenv("USER", "appuser")
group = os.getenv("GROUP", "appuser")

# Обработка сигналов
raw_env = [
    "ENVIRONMENT=production",
    f"MODEL_PATH={os.getenv('MODEL_PATH', '/app/models/final_model.pkl')}",
    f"SCALER_PATH={os.getenv('SCALER_PATH', '/app/models/scaler.pkl')}"
]

# Хуки
def pre_fork(server, worker):
    pass

def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def when_ready(server):
    server.log.info("Server is ready. Spawning workers")

def worker_int(worker):
    worker.log.info("Worker received INT or QUIT signal")

def worker_abort(worker):
    worker.log.info("Worker received SIGABRT signal")