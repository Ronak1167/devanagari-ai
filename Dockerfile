FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
# We use the CPU-only version of PyTorch to keep the image size small
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 7860 (Hugging Face Spaces requirement)
EXPOSE 7860

# Command to run the application using Gunicorn
# Hugging Face routes traffic to port 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "wsgi:app"]
