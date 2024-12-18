# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /

# Copy the pyproject.toml and poetry.lock files into the container
COPY . .

# Install Poetry
RUN pip install --no-cache-dir poetry

# Install the dependencies using Poetry
RUN poetry config virtualenvs.create false && poetry install --no-dev

# Install necessary packages
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
ARG SAS_TOKEN
# Download the vector database
RUN azcopy cp "https://taicdocumentsearcherdata.blob.core.windows.net/vectordb/prod/*?$SAS_TOKEN" vectordb --recursive

# Expose the port that the app runs on
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]