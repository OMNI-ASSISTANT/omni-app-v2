FROM python:3.12-slim   # Start with official Python image
WORKDIR /app            # cd /app
COPY requirements.txt . # bring in deps list
RUN pip install -r requirements.txt  # bake them in
COPY . .                # copy your code
CMD ["python", "omni.py"]  # run it when the container starts 