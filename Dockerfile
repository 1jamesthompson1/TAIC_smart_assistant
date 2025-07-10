# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Create a non-root user with a home dir
RUN adduser --disabled-password --gecos '' appuser

# Set working dir and permissions
WORKDIR /app
COPY . .
RUN mkdir -p /app/uv-cache && chown -R appuser:appuser /app/uv-cache

# Install uv
RUN pip install --no-cache-dir uv

# Create a world-writable cache directory for uv and set it for uv
RUN mkdir -p /app/uv-cache && chmod -R 777 /app/uv-cache
ENV UV_CACHE_DIR=/app/uv-cache

# Install the dependencies using uv
RUN uv sync --no-dev

# Install necessary packages to allow download of azcopy
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    apt-transport-https

# Download and install the Microsoft package repository
RUN curl -sSL -O https://packages.microsoft.com/config/debian/11/packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    rm packages-microsoft-prod.deb && \
    apt-get update

# Install azcopy
RUN apt-get install -y azcopy

# Download the vector database
RUN --mount=type=secret,id=SAS_TOKEN,mode=0444,required=true \
    azcopy cp "https://taicdocumentsearcherdata.blob.core.windows.net/vectordb/prod/*?$(cat /run/secrets/SAS_TOKEN)" vectordb --recursive

# Expose the port that the app runs on
EXPOSE 7860

# Command to run the application
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
