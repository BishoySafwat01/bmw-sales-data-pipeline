# BMW Global Sales: Fault-Tolerant Data Pipeline & Predictive AI Platform

An automated multi-stage Data Quality Assurance (QA) and cleaning pipeline engineered to handle high-dimensional, corrupted enterprise automotive datasets. The system features a robust statistical simulation layer to validate database resilience and deploys a predictive intelligence backend to forecast used car prices in real time.

## Data Pipeline & Workflow

1. **Corruption Simulation:** A production-grade Simulation Module uses statistical methods to inject real-world system noise (structured textual noise, 5% duplication patterns, and 1.5% missing value vectors).
2. **Data QA & Sanitization:** Implements deterministic parsing and imputation logic using dynamic aggregation metrics (statistical modes and conditional medians) alongside 3.0 IQR Winsorization to eradicate heavy outliers without data loss.
3. **Feature Engineering:** Executes optimal Log-Transformations to eliminate feature skewness, scales features via Scikit-Learn standard normalization, and applies customized One-Hot Matrix Encoding.
4. **Predictive Scoring & UI:** Integrates the predictive asset into an interactive Streamlit Web Application for real-time used car price estimation through an aligned AI scoring backend.

## Core Technical Features

* **Automated Data QA:** Successfully resolved system data corruption patterns across thousands of high-dimensional records without generating training bias.
* **Statistical Outlier Eradication:** Engineered robust 3.0 IQR Winsorization filters to clip extreme statistical variances while preserving dataset integrity.
* **Production Deployment:** Deployed the final predictive analytics model into a responsive dashboard framework utilizing Power BI metrics, Seaborn visualizations, and a live Streamlit scoring interface.

## Technical Stack

* **Languages & Analytics:** Python, Pandas, NumPy, Joblib
* **Machine Learning:** Scikit-Learn (Predictive Modeling, Normalization)
* **Visualization & Dashboards:** Streamlit, Power BI, Matplotlib, Seaborn, Git

---

## Local Setup & Deployment

### Steps
1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/BishoySafwat01/bmw-sales-data-pipeline.git](https://github.com/BishoySafwat01/bmw-sales-data-pipeline.git)
   cd bmw-sales-data-pipeline
2. **Install Dependencies:**
   Execute the pip pipeline matrix to install all the required pre-requisites and libraries at once:
   ```bash
   pip install -r requirements.txt
3. **Launch Streamlit Web App:**
   Spin up the local production-grade host to interface with the predictive engine inside your browser:
    ```bash
    streamlit run app.py
