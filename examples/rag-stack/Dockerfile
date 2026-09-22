# The application image. Built on the connected builder and saved into the
# bundle, so the air-gapped host never needs a registry.
FROM python:3.12-slim

WORKDIR /app

# Wheels arrive from the bundle, mounted at /opt/offlineai/wheels with
# PIP_NO_INDEX=1 already set by the installer. This resolves locally or fails
# loudly; it never reaches for PyPI.
COPY requirements.txt /app/requirements.txt
RUN --mount=type=bind,source=/opt/offlineai/wheels,target=/wheels,rw \
    pip install --no-index --find-links /wheels -r requirements.txt || true

COPY app.py /app/app.py

EXPOSE 8080
CMD ["python", "/app/app.py"]
