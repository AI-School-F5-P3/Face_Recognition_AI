# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements/base.txt requirements/base.txt
COPY requirements/prod.txt requirements/prod.txt
RUN pip install --no-cache-dir -r requirements/prod.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/employee_database \
    /app/data/trained_models \
    /app/data/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV MONGODB_URL=mongodb://mongodb:27017
ENV OPENCV_VIDEOIO_PRIORITY_MSMF=0

# Run the application
CMD ["python", "main.py"]