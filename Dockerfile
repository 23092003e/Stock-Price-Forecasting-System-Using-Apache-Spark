# Dockerfile
FROM apache/spark-py:3.4.0

# Set environment variables
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Create and set working directory
WORKDIR /app

# Copy the application code
COPY . .

# Command to run the application
CMD ["python3", "lstm.py"]