FROM registry.access.redhat.com/ubi9/python-312

# Set up a working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Command to run the application
CMD ["python", "image-builder-mcp.py"]
