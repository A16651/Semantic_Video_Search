# Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    # libgl1 needed for opencv-python we use opencv-python-headless
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make the wrapper script executable
RUN chmod +x ./wait_for_quadrant.sh

# Expose port
EXPOSE 8000

# Run the application
CMD ["./wait_for_quadrant.sh", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
