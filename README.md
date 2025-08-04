# IBM-Cloud-Automated-Threat-Detection-NIDS
# Automated Threat Detection using Machine Learning on IBM Cloud

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)
![IBM Cloud](https://img.shields.io/badge/IBM%20Cloud-%230530AD.svg?logo=ibm&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?logo=Jupyter&logoColor=white)

A robust Network Intrusion Detection System (NIDS) designed to automatically identify and classify various types of cyberattacks from network traffic data. This project leverages a machine learning pipeline on the IBM Cloud platform to provide an effective early-warning system against malicious network activities.

## 🚀 Project Overview

In today's interconnected world, network security is paramount. This project addresses the critical challenge of detecting cyber threats in real-time by building an intelligent system that can distinguish between normal network activity and malicious attacks. The system is trained on the KDD'99 dataset to recognize patterns associated with various attack categories, including **Denial-of-Service (DoS)**, **Probe**, **Remote-to-Local (R2L)**, and **User-to-Root (U2R)**.

The core of this project is a complete data science workflow, from data preprocessing and exploratory analysis to model training and evaluation, culminating in the use of IBM's cutting-edge **AutoAI** technology to discover the optimal model pipeline.

---

## ✨ Key Features

* **Multiclass Classification:** Goes beyond simple anomaly detection by classifying attacks into four specific, high-risk categories.
* **Advanced Data Preprocessing:** Implements a robust `scikit-learn` pipeline to handle both numerical and categorical data, ensuring a clean and scalable workflow.
* **Intelligent Model Training:** Utilizes a `RandomForestClassifier` for a powerful and interpretable baseline model.
* **Automated Model Building with IBM AutoAI:** Leverages IBM's AutoAI service to automatically generate, train, and rank numerous high-performance model pipelines, ensuring the best possible model is selected.
* **Feature Importance Analysis:** Identifies the key network features that are most predictive of an attack, providing valuable insights for security analysts.
* **Cloud-Native Architecture:** Fully developed and deployed using essential IBM Cloud services, including Watson Studio and Cloud Object Storage.

---

## 🛠️ Technology Stack

* **Cloud Platform:** IBM Cloud (Watson Studio, Cloud Object Storage, `watsonx.ai`)
* **Primary Language:** Python 3.11
* **Core Libraries:**
    * `pandas` & `numpy` for data manipulation
    * `scikit-learn` for machine learning pipelines and modeling
    * `matplotlib` & `seaborn` for data visualization

---

## ⚙️ Project Workflow

The project follows a structured machine learning lifecycle:

1.  **Data Ingestion:** The KDD'99 dataset is loaded from IBM Cloud Object Storage.
2.  **Exploratory Data Analysis (EDA):** The data is analyzed to understand the distribution of normal traffic vs. different attack classes. In this project, multiclass labels were simulated to demonstrate the model's full capabilities.
3.  **Data Preprocessing:** A `ColumnTransformer` pipeline is used to apply standard scaling to numerical features and one-hot encoding to categorical features.
4.  **Model Training & Evaluation:**
    * A baseline `RandomForestClassifier` is trained on the preprocessed data.
    * The model's performance is evaluated using accuracy, a detailed classification report, and a confusion matrix.
5.  **Automated Machine Learning:**
    * The preprocessed data is used as input for an **IBM AutoAI experiment**.
    * AutoAI automatically generates and evaluates multiple model pipelines, identifying the top-performing algorithm.
6.  **Model Selection:** The best model from the AutoAI leaderboard is saved, ready for deployment.

---

## 📊 Results & Performance

The manually built Random Forest model achieved a high accuracy score, demonstrating strong performance in identifying and classifying threats. The AutoAI experiment further improved upon this baseline, discovering an optimized pipeline with even greater precision and recall.
## ✅ Results

| Model                    | Accuracy   |
|--------------------------|------------|
| Random Forest (Manual)   | 86.3%      |
| Batched Tree Ensemble    | **88.3%**  |

> The **Batched Tree Ensemble** classifier from IBM AutoAI outperformed manual models.

---<img width="2049" height="1065" alt="Screenshot 2025-08-04 at 1 35 16 AM" src="https://github.com/user-attachments/assets/42d6dff6-0f96-444c-b598-db10d3c64998" />
<img width="2049" height="1067" alt="Screenshot 2025-08-04 at 1 35 36 AM" src="https://github.com/user-attachments/assets/b689ed8f-1788-4f3d-b993-cab93a1ac5b5" />
<img width="2052" height="1060" alt="Screenshot 2025-08-04 at 1 36 02 AM" src="https://github.com/user-attachments/assets/04dcc081-1b06-421a-a6ee-a975b4a24ef0" />

---

## 🚀 How to Run this Project

To replicate this project, follow these steps:

1.  **Set up IBM Cloud:**
    * Create a free IBM Cloud account.
    * Provision **Watson Studio** and **Cloud Object Storage** services.
2.  **Create a Project:**
    * In Watson Studio, create a new empty project.
    * Associate the Cloud Object Storage instance with your project.
3.  **Upload Assets:**
    * Upload the `Final_Working_NIDS.ipynb` notebook and the `Train_data.csv` & `Test_data.csv` datasets to the project's **Assets** page.
4.  **Run the Notebook:**
    * Open the notebook and select a Python 3.11 runtime environment.
    * Run all the cells from top to bottom. This will perform the analysis and generate the `Train_data_multiclass_for_autoai.csv` file needed for the next step.
5.  **Run the AutoAI Experiment:**
    * From the Assets page, create a **New AutoAI asset**.
    * Select `Train_data_multiclass_for_autoai.csv` as the data source.
    * Choose **`attack_category`** as the column to predict.
    * Run the experiment and save the best-performing model.

---

## 📈 Future Improvements

* **Real-time Deployment:** Deploy the saved model as a web service to create a live API endpoint for real-time network traffic analysis.
* **Advanced Deep Learning:** Experiment with deep learning models like LSTMs or Transformers, which can capture sequential patterns in network traffic.
* **Scalability:** Integrate the solution with big data tools like Apache Spark to handle massive-scale network data streams.
