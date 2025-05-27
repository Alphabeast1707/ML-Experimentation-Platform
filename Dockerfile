# Use a base image appropriate for your application
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port your application runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "app.py"]
