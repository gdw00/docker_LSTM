# Use NVIDIA CUDA base image (compatible and available version)
# FROM nvidia/cuda:11.8.0-base
FROM nvcr.io/nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Copy dependency file
COPY requirements.txt ./

# Install dependencies
RUN apt-get update && apt-get install -y python3 python3-pip && \
    pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# Copy the entire project
COPY . .

# Set environment variables for data and model paths
ENV DATA_PATH="/app/data"
ENV MODEL_SAVE_PATH="/app/models"

# Create model output directory
RUN mkdir -p $MODEL_SAVE_PATH

# Run the training script
CMD ["python3", "train_LSTM.py"]
