# **LSTM Training Workflow: Model Training and Evaluation**

This repository contains code for training an LSTM-based model using climate and weather-related datasets.



## **Download the Dataset**
1. Download the dataset from [Google Drive](https://drive.google.com/file/d/1bZFN7HIQ1h-wYX830gIu7VbZu8Zo4HKd/view?usp=sharing).
2. Extract and place the dataset into the `./dataset` directory.


## **Run the Training Pipeline**

### **1. Build the Docker Image**
```bash
docker build -t lstm_training_image .
```

### **2. Run the Docker Container**
```bash
docker run --gpus all -it \
    -v ./dataset:/app/data \
    -v ./models:/app/models \
    -v ./results:/app/results \
    lstm_training_image
```






