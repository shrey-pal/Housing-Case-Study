# ==============================================================================
# Dockerfile
# ==============================================================================
# Use a slim and recent Python version as the base image
FROM python:3.9-slim-buster

# Set the working directory inside the container

WORKDIR /app

# Copy the requirements file first to take advantage of Docker's layer caching

COPY requirement.txt requirement.txt

# Install the Python dependencies

RUN pip install --no-cache-dir -r requirement.txt

# Copy the rest of your application files into the container

# This includes app.py, your .pkl files, and the 'templates' folder

COPY . .

# The PORT environment variable is automatically set by hosting services like Render.

# This line makes the container listen on that port.

EXPOSE $PORT

# Command to run the application using Gunicorn, a production-ready web server.

# It binds to all network interfaces on the port provided by the environment.

CMD ["gunicorn", "--workers=4", "--bind", "0.0.0.0:$PORT", "app:app"]

# \==============================================================================


