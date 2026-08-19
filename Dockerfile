# CPU image: the whole system, reproducibly, with no GPU and no local setup.
#
#   docker build -t hetero-serve .
#   docker run --rm hetero-serve                      # run the test suite
#   docker run --rm hetero-serve python run_demo.py --model tiny
#   docker run --rm -p 8000:8000 hetero-serve \
#       python -m heteroserve.dashboard.server --model tiny --devices CPU,CPU
#
# Everything runs on the numpy reference engine here. That is the point: the
# scheduler, the paged KV cache, the migration path and the network shaper are
# all device-independent, so they work identically with no accelerator present.
# For the CUDA kernels see Dockerfile.cuda.

FROM python:3.12-slim

# Compilers are not needed for the CPU path, but git is, so the image can be
# used to clone and diff the repo when debugging inside a container.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch comes from the CPU index: the full CUDA wheel is ~2.5 GB and buys
# nothing without a GPU.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.4"
RUN pip install --no-cache-dir \
        "numpy>=2.0" \
        "regex>=2024.0" \
        "pytest>=8.0" \
        "matplotlib>=3.8" \
        "fastapi>=0.110" \
        "uvicorn>=0.27"

COPY . /app

# GPT-2 weights are ~550 MB and deliberately not baked in — `run_demo.py`
# fetches them on first use. Run with `-v hetero_weights:/app/weights` to keep
# them across containers.
VOLUME ["/app/weights"]

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

# Default: prove the thing works. Tests needing GPT-2 weights or CUDA skip with
# a reason rather than failing.
CMD ["python", "-m", "pytest", "-q"]
