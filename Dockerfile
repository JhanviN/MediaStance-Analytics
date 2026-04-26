FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements_hf.txt .
RUN pip install --no-cache-dir -r requirements_hf.txt

# Copy app code
COPY . .

# Create necessary directories
RUN mkdir -p data models/transformer_bilateral logs

# Expose Streamlit port (HF Spaces uses 7860)
EXPOSE 7860

# Run Streamlit
CMD ["streamlit", "run", "demo/demo_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
