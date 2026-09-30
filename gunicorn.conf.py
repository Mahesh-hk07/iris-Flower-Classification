"""
Gunicorn configuration file for FloraVision AI 2.0.
Auto-loaded by Gunicorn on startup to ensure stable, single-threaded PyTorch execution
in resource-constrained cloud containers (e.g. Render 512MB RAM free tier).
"""

import os

# Limit CPU threads to avoid OpenMP deadlock and CPU thrashing in containers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = 1
threads = 2
timeout = 120
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = "info"
preload_app = False
