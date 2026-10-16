FROM python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /workspace
COPY requirements.txt requirements-docker.txt requirements-docker.lock /opt/requirements/
RUN pip install --no-cache-dir -r /opt/requirements/requirements-docker.txt -c /opt/requirements/requirements-docker.lock \
    && pip check
CMD ["bash"]
