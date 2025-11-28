# EcoSort: End-to-End Waste Classification Pipeline

## Project Description
EcoSort is a Machine Learning pipeline designed to automate waste segregation. Building on the principles of sustainability (extending the agricultural use case), this project leverages Deep Learning (MobileNetV2) to classify waste items into **Recyclable (Paper)**, **Organic (Rock)**, and **Hazardous (Scissors)** categories using non-tabular image data.

The system includes a full MLOps lifecycle:
1.  **Data Ingestion:** Automated handling of image datasets.
2.  **Model Training:** Transfer learning with MobileNetV2.
3.  **Deployment:** Interactive Streamlit UI via Cloud Tunneling.
4.  **Retraining Loop:** A trigger system to process new user-uploaded data.

## GitHub Repository
https://github.com/AgbajeCity/eco-sort-pipeline

## Directory Structure
The project adheres to the following structure:
eco-sort-pipeline/ │ ├── README.md # Project documentation and setup │ ├── notebook/ │ └── EcoSort_Project.ipynb # Training logic and evaluation metrics │ ├── src/ │ ├── preprocessing.py # Image transformation logic │ ├── model.py # MobileNetV2 architecture definition │ └── prediction.py # Inference logic │ ├── data/ │ ├── train/ # Training images │ └── test/ # Validation/Testing images │ └── models/ └── waste_model.h5 # Trained TensorFlow model

## Setup Instructions
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/AgbajeCity/eco-sort-pipeline.git](https://github.com/AgbajeCity/eco-sort-pipeline.git)
    cd eco-sort-pipeline
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

## Video Demo
https://youtu.be/HRANoPmcuWg
*This video demonstrates the prediction process, visualization interpretations, and the retraining trigger.*

## Flood Request Simulation (Results)
To satisfy the scalability requirement, we performed a load test using **Locust** with the following parameters:
* **Users:** 50 Concurrent Users
* **Spawn Rate:** 5 users/second
* **Duration:** 10 seconds

**Performance Metrics:**
* **Average Latency:** ~45 ms
* **Failure Rate:** 0%
* **Requests Per Second (RPS):** ~40.5

*The model demonstrated stability under high load, serving predictions with minimal latency.*

## Model Evaluation
The model was evaluated using Accuracy, Precision, Recall, and F1-Score (see Notebook for detailed classification report).
* **Accuracy:** ~96%
* **Optimization:** Used Transfer Learning (MobileNetV2) and Early Stopping to prevent overfitting.
