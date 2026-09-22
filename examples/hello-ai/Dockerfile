# Built on the connected builder, saved into the bundle, and loaded on the
# air-gapped target. Nothing here reaches the network at run time.
FROM python:3.12-slim

WORKDIR /app
COPY app.py /app/app.py

EXPOSE 8000
CMD ["python", "/app/app.py"]
