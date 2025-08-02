# ==============================================================================
# Dockerfile
# ==============================================================================
# Use a slim Python version for a smaller image size
FROM python:3.9-slim-buster

# Set the working directory inside the container

WORKDIR /app

# Set a default port. Render will override this with its own port.

ENV PORT 10000

# Expose the port the app will run on

EXPOSE 10000

# Copy the requirements file first to leverage Docker's layer caching

COPY requirement.txt requirement.txt

# Install the dependencies

RUN pip install --no-cache-dir -r requirement.txt

# Copy the rest of your application files

COPY . .

# Command to run the application using Gunicorn

CMD ["gunicorn", "--workers=4", "--bind", "0.0.0.0:$PORT", "app:app"]

# \==============================================================================

# requirements.txt

# \==============================================================================

# I've added specific versions to make your build more reliable.