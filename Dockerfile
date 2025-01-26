# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy language model
RUN python -m spacy download en_core_web_sm

# Expose the port the Flask app runs on
EXPOSE 5000

# Command to run the Flask application
CMD ["python", "app.py"]
