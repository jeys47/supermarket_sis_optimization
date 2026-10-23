#  Smart Layout & Retail Analytics: Supermarket Spatial Incoherence Score (SIS) Optimization

An end-to-end Data Science and Prescriptive Analytics solution designed to optimize supermarket spatial layout (*Store Layout*). By combining **Non-Supervised Machine Learning (K-Means Clustering)**, **Association Rule Mining (FP-Growth)**, and **Financial Pareto Analysis (ABC)**, this project identifies high-value cross-selling pairs and minimizes customer transit friction using the custom **Shop-In-Shop (SIS) Incoherence Index**.

---

##  Executive Summary & Core Concept

In physical retail, poor aisle layout causes friction in the customer journey and leads to missed sales opportunities. Unlike standard predictive models that estimate future demand, this project focuses on **Prescriptive Analytics & Operational Research**:

1. **Who buys?** Segmenting the customer base into 4 behavioral archetypes.
2. **What sells together?** Extracting strong cross-selling rules across departments.
3. **Is it profitable?** Validating that reallocations focus on **Class A (70% Revenue)** departments.
4. **Where should departments be placed?** Optimizing physical layout by minimizing physical distance between highly affine departments (SIS Index).

---

##  Methodology & Key Features

### 1. Customer Behavioral Segmentation (K-Means)
A K-Means clustering model trained on RFM (Recency, Frequency, Monetary) and Promotional sensitivity metrics to group clients into 4 distinct segments:
* **Cluster 0: Champions / VIP** (45.59%) – High spenders, high purchase frequency.
* **Cluster 1: Inactive / At Risk** (13.31%) – Low frequency, high recency.
* **Cluster 2: Occasional / Standard** (25.03%) – Average spending and basket sizes.
* **Cluster 3: Deal Hunters** (16.07%) – Highly responsive to discounts and promotions.

### 2. Cross-Selling & Association Rules (FP-Growth)
Extraction of co-purchase patterns across transactions to discover high-affinity product pairs (e.g., *Dry Grocery + Sauces/Soups*, *Canned Goods + Home Care*) using Support, Confidence, and Lift.

### 3. Financial ABC / Pareto Validation
Prioritization of reallocations using the Pareto principle to ensure that layout modifications target top-performing Class A departments generating over 70% of store turnover.

### 4. Spatial SIS Incoherence Scoring
Quantitative evaluation of store efficiency using the custom **Shop-In-Shop (SIS) Index**:
$$\text{SIS Index} = \text{Lift} \times \text{Aisle Distance}$$
Higher scores highlight severe spatial incoherence (high affinity but physically separated), signaling immediate candidates for layout restructuring.

---

##  Repository Structure

```text
.
├── app.py                                  # Interactive Streamlit Web Application
├── requirements.txt                        # Project dependencies
├── README.md                               # Project documentation and setup guide (this file)
├── LICENSE                                 # Official MIT License file containing legal terms of use
├── .gitignore                              # Specifies intentionally untracked files to ignore (e.g., venv)
│
│
├── Notebooks (Pipeline & Models)
│   ├── nettoyage.ipynb                     # Data cleaning and preprocessing pipeline
│   ├── eda_final.ipynb                     # Exploratory Data Analysis & Pareto/ABC charts
│   ├── cluster.ipynb                       # K-Means training & RFM customer profiling
│   ├── data_mining_fpgrowth.ipynb          # FP-Growth association rules extraction
│   ├── distance_sis.ipynb                  # Spatial layout & SIS incoherence calculations
│   ├── moteur_recommandation.ipynb         # Layout recommendation engine
│   └── validation.ipynb                    # Financial validation & scenario tests
│
├── Python Utility Scripts
│   ├── check_data.py                       # Data integrity and schema verification
│   ├── eda.py                              # Automated EDA execution script
│   └── eda_test.ipynb                      # Testing notebook for EDA functions
│
└── Datasets & Outputs
    ├── supermarket_sales_100k.csv          # Raw transaction dataset (100k records)
    ├── dataset_cleaned.csv                 # Processed & cleaned master dataset
    ├── clients_segmentes.csv               # Segmented client base with assigned clusters
    ├── regles_association_rayons.csv       # Extracted association rules and affinity scores
    └── score_incoherence_spatiale_sis_corrige.csv # Computed SIS spatial scores

```

---

##  Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jeys47/supermarket_sis_optimization.git
   cd supermarket_sis_optimization
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

##  Running the Streamlit Web Application

To launch the interactive dashboard and explore layout recommendations, customer clusters, and financial simulations:

```bash
streamlit run app.py
```

The application features:
* **Segment Analysis Tab:** Interactive visualization of K-Means customer profiles and RFM statistics.
* **Affinity & SIS Tab:** Heatmaps of department affinity, co-purchases, and spatial incoherence scores.
* **Layout Optimization Simulator:** Dynamic "What-If" scenario tool for supermarket managers.

---

##  Tech Stack

* **Language:** Python 3.9+
* **Data Manipulation & Mining:** `pandas`, `numpy`, `mlxtend` (FP-Growth)
* **Machine Learning:** `scikit-learn` (K-Means Clustering)
* **Visualization:** `matplotlib`, `seaborn`, `plotly`
* **Web Interface:** `streamlit`

---

##  License

Distributed under the MIT License. See `LICENSE` for more information.
```
eof