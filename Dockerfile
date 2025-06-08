# Use an official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only needed files
COPY app.py .
COPY utils.py .
COPY data/ ./data/

# If embeddings_index.pkl exists, copy it (this is a trick — optional file)
# Use a wildcard match; Docker won't fail if no match
COPY embeddings_index.pkl* ./

# Expose a port if you later serve it via Flask/FastAPI (optional)
# EXPOSE 8000

# Default command (example: run app.py)
CMD ["python", "app.py"]