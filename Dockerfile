# Start with official slim Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Default command
CMD ["python", "omni.py"]
