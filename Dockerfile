# ==============================================================================
# Dockerfile
# ==============================================================================
# Use a slim Python version for a smaller image size
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Set a default port. Render will override this with its own port.
ENV PORT 10000

# Copy the requirements file first
COPY requirements.txt requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application files
COPY . .

# Expose the port the app will run on
EXPOSE 10000

# Command to run the application using Gunicorn.
# This format allows the $PORT variable to be correctly read by the hosting service.
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app
