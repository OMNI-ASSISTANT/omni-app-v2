# Start with official slim Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
# Copy the rest of your code
COPY . .

# Default command
CMD ["python", "omni.py", "start"]
