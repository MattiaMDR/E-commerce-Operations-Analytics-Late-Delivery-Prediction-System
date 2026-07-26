
# E-commerce Operations Analytics & Late-Delivery Prediction System

## Project Overview

This project presents an end-to-end e-commerce operations analytics and machine learning workflow designed to identify orders at risk of late delivery and support operational decision-making.

The project combines data preparation, exploratory data analysis, feature engineering, predictive modeling, threshold evaluation and business intelligence reporting.

Python and scikit-learn were used to build and compare machine learning models, while Tableau was used to transform the results into an operational decision dashboard.

---

## Business Problem

Late deliveries can negatively affect customer satisfaction, operational efficiency and business performance.

The purpose of this project was to develop an early-warning system capable of identifying potentially late orders before delivery and helping an operations team answer questions such as:

- Which orders should be reviewed first?
- Which shipping modes have the highest late-delivery rates?
- How much revenue is associated with orders flagged for review?
- Which regions present the highest operational exposure?
- How does changing the prediction threshold affect workload and late-order detection?
- What balance should be used between operational capacity and risk coverage?

The final solution was designed as a decision-support system rather than only a technical prediction model.

---

## Dataset

The original data contained:

- **180,519 transaction-level records**
- Multiple rows associated with the same customer order
- Product, customer, shipping, geographic and financial information

The transaction-level records were consolidated into:

- **65,752 unique orders**

This order-level aggregation prevented duplicated orders from distorting operational KPIs and machine learning results.

The repository includes:

data/dataco_orders.csv

dataco_orders.csv is the consolidated order-level dataset used for the main analysis and machine learning workflow.

Some original, intermediate and generated datasets are not included because they exceeded the recommended GitHub file-size limits.

Data Availability

The repository contains the main order-level dataset required to understand and reproduce the project workflow.

The following larger files may not be included:

Original transaction-level dataset
Cleaned intermediate datasets
Engineered model-ready datasets
Prediction output files
Threshold comparison exports
Tableau data-source exports

These files were excluded because of their size and because most of them can be recreated by running the Python scripts included in the repository.

To reproduce the project, place dataco_orders.csv inside the data/ folder and execute the scripts in the order described in the workflow section.

Project Workflow

The project was developed through the following stages:

1. Data Inspection and Validation

The first stage examined:

Dataset structure
Column types
Missing values
Duplicate records
Order identifiers
Date fields
Shipping information
Revenue-related variables
Potential data-quality issues
2. Order-Level Aggregation

The original dataset contained multiple product rows for individual orders.

The data was therefore consolidated at order level to create one observation per unique order.

This stage included:

Order-level revenue aggregation
Product quantity aggregation
Customer and shipping information consolidation
Delivery-status validation
Date-field preparation
3. Exploratory Data Analysis

Exploratory analysis was performed to understand:

Overall late-delivery performance
Late-delivery rate by shipping mode
Regional operational differences
Revenue exposure
Delivery-risk patterns
Distribution of target classes
Relationships between order characteristics and delivery outcomes
4. Feature Engineering

Relevant features were prepared for machine learning while avoiding variables that could introduce target leakage.

The feature-engineering process included:

Date-based features
Order-value information
Shipping attributes
Customer and geographic characteristics
Categorical-variable encoding
Missing-value treatment
Numeric-variable preprocessing
5. Time-Based Validation

A time-based train/test split was used instead of a random split.

This approach better represents a real operational scenario in which historical orders are used to predict the outcomes of later orders.

The validation strategy also reduced the risk of obtaining unrealistically optimistic results from information shared across time periods.

6. Machine Learning Model Comparison

The following classification models were evaluated:

Baseline model
Logistic Regression
Random Forest
HistGradientBoosting

The models were compared using classification and ranking metrics, with particular attention to their usefulness for operational decision-making.

7. Threshold Analysis

The default probability threshold was not accepted automatically.

Different thresholds were analyzed to measure the trade-off between:

Late orders detected
Orders flagged for review
Operational workload
Precision
Recall
Revenue exposure
False-positive volume

Two operational approaches were considered:

30% threshold: broader early-warning review
40% threshold: more focused operational review
8. Tableau Dashboard Development

The final model results and operational metrics were exported to Tableau.

The dashboard was designed to communicate the analysis clearly to business and operations users.

Machine Learning Results

Logistic Regression was selected as the final model because it provided a strong balance between predictive performance, interpretability and operational usability.

Main results
ROC-AUC: 0.7447
Selected early-warning threshold: 30%
Actual late orders identified: 89.96%
Model-validation method: time-based split

The final model was not selected only by comparing accuracy scores.

The decision also considered:

Ability to rank risky orders
Detection of actual late orders
Operational review volume
Model interpretability
Threshold flexibility
Business usability
Tableau Dashboard

The final Tableau dashboard is called:

Operations Decision Dashboard

The dashboard includes:

Total orders analyzed
Overall late-delivery rate
Orders flagged for review
Late orders detected
Revenue flagged for operational review
Late-delivery rate by shipping mode
Predicted risk-level distribution
Revenue flagged by region
Operational review queue by threshold

The dashboard helps users compare operational workload and delivery-risk coverage under different review thresholds.

Dashboard Preview

Add the final dashboard image inside the images/ or dashboard/ folder and update the path below if necessary.


After uploading the image, remove the code block markers around the line above so GitHub displays the image directly.

depending on where you save the screenshot.

Key Business Insights

The project demonstrates that model performance must be translated into operational decisions.

A lower probability threshold identifies a larger share of actual late orders, but it also increases the number of orders requiring review.

A higher threshold reduces workload but may fail to identify additional late orders.

The selected 30% threshold was used as an early-warning setting because it identified 89.96% of actual late orders during the evaluation period.

The dashboard also highlights differences across:

Shipping modes
Geographic regions
Predicted risk categories
Revenue exposure
Review strategies

These results allow an operations team to prioritize orders based on both predicted risk and business impact.

Risk-Level Segmentation

Predicted probabilities were grouped into operational risk levels to make model results easier to interpret.

The segmentation supports a review process in which orders can be categorized as:

Low risk
Watch
High risk
Critical risk

This makes the model output more practical than presenting raw prediction probabilities alone.

Repository Structure

Update the Python filenames below so they match the real files in your repository.

Some output files shown above may not be stored in the repository because of file-size limitations. They can be generated by running the project scripts.

Python Scripts

The Python scripts separate the project into clear analytical stages.

Their responsibilities include:

Loading and validating the dataset
Cleaning and preparing order-level data
Performing exploratory analysis
Creating model features
Training and comparing classification models
Evaluating prediction thresholds
Exporting results for Tableau

Run the scripts according to their numerical or logical order.

For example:

python notebooks/01_data_understanding_cleaning.py
python notebooks/02_operations_kpi_analysis.py
python notebooks/03_build_order_level_dataset.py
python notebooks/04_order_level_kpi_analysis.py
python notebooks/05_late_delivery_prediction.py
python notebooks/06_model_comparison_and_threshold_analysis.py
python notebooks/07_tableau_decision_dataset.py

Replace these commands with the actual filenames used in the repository.

How to Run the Project
1. Clone the repository
git clone [https://github.com/YOUR-USERNAME/ecommerce-late-delivery-prediction.git](https://github.com/MattiaMDR/ecommerce_operations_analytics_decision_system)
2. Open the project folder
cd ecommerce-late-delivery-prediction
3. Create a virtual environment

On Windows:

python -m venv .venv
.venv\Scripts\activate

On macOS or Linux:

python3 -m venv .venv
source .venv/bin/activate
4. Install the required libraries
pip install -r requirements.txt
5. Confirm the dataset location

Make sure the following file is available:

data/dataco_orders.csv
6. Run the analysis

Execute the Python scripts in the correct project order or open the Jupyter Notebook:

jupyter notebook

Then open:

notebooks/ecommerce_operations_analysis.ipynb
Requirements

The main technologies and libraries used were:

Python
Pandas
NumPy
scikit-learn
Matplotlib
Jupyter Notebook
Tableau

A basic requirements.txt file may contain:

pandas
numpy
matplotlib
scikit-learn
jupyter

Only add additional packages if they are actually imported by the project scripts.

Tools and Skills Demonstrated
Data and Programming
Python
Pandas
NumPy
Data Cleaning
Data Validation
Data Aggregation
Exploratory Data Analysis
Feature Engineering
Machine Learning
Predictive Analytics
Binary Classification
Logistic Regression
Random Forest
Gradient Boosting
Model Evaluation
ROC-AUC
Precision and Recall
Threshold Optimization
Time-Based Validation
Data-Leakage Prevention
Business Intelligence
Tableau
Dashboard Design
Data Visualization
KPI Development
E-commerce Analytics
Operations Analytics
Delivery-Risk Analysis
Revenue Exposure Analysis
Decision Support
Project Limitations

This project was developed as a portfolio case study using historical data.

The predictions should not be interpreted as a production deployment without additional validation.

A real production implementation would also require:

Continuous data-quality monitoring
Model-drift monitoring
Periodic model retraining
Integration with order-management systems
Real-time or scheduled prediction pipelines
Operational feedback from reviewed orders
Cost-sensitive threshold calibration

The available variables also determine the maximum predictive performance that can be achieved.

Additional information such as carrier events, warehouse capacity, traffic conditions, weather data and real-time logistics updates could improve future model performance.

Possible Future Improvements

Potential extensions include:

Cost-sensitive model evaluation
Probability calibration
SHAP-based model explanations
Carrier-level performance analysis
Automated model monitoring
Real-time order scoring
API deployment
Integration with Power BI or Tableau Server
Review-capacity optimization
Additional logistics and external data sources

These improvements are not required for the current portfolio version but represent possible production-level extensions.

Project Purpose

This project was developed to demonstrate how raw e-commerce data can be transformed into:

Reliable order-level information
Predictive risk scores
Operational review strategies
Business KPIs
Interactive decision-support reporting

The main objective was not only to build a machine learning model, but to connect predictive analytics with practical e-commerce operations decisions.

Author

Mattia 

E-commerce Data Analyst focused on:

E-commerce analytics
Machine learning
Predictive analytics
Inventory and operations analysis
Python and SQL
Tableau and Power BI dashboards

