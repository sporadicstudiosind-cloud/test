# Iridium-1 inference container.
#
# CPU-only torch on purpose: the whole point of the small rungs is that the
# architecture runs without an accelerator, and a CUDA image would be 6 GB to
# no benefit on the hardware this is deployed to.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    IRIDIUM_ENABLE_1B=1 \
    PORT=8080

WORKDIR /app

# torch first so the heavy layer caches across code changes.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
 && pip install numpy pyyaml

COPY iridium/ /app/iridium/
COPY serve/ /app/serve/

# Fail the build rather than the first request if the weights did not copy.
RUN python -c "import torch, pathlib; p=pathlib.Path('/app/serve/weights/nano-phase1-fp16.pt'); \
    assert p.exists(), 'checkpoint missing from image'; print('checkpoint', p.stat().st_size//1024//1024, 'MB')" \
 && python -c "from iridium.config import get_config; c=get_config('test1b'); print('config ok:', f'{c.n_params:,}', 'params')"

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8080)}/api/health', timeout=4)"

CMD ["python", "serve/server.py"]
