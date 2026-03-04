Silent Customer Churn Detection System

Project Overview

The Silent Customer Churn Detection System is a machine learning-based predictive analytics project designed to identify customers who are likely to churn without explicitly informing the business.
Unlike traditional churn cases where customers cancel subscriptions, silent churners gradually stop purchasing or interacting with the business. This system detects such behavior early using transaction history and behavioral patterns.

Objective

Identify customers at risk of silent churn
Assign a Risk Score to each customer
Enable proactive retention strategies
Support business decision-making with data-driven insights


Dataset Used

Online Retail Transaction Dataset
Features include:
         CustomerID
         Invoice Date
         Quantity
         Unit Price
         Country
         Transaction Amount


Project Workflow

1️ Data Collection
Imported transaction dataset using Pandas

2️ Data Cleaning
Removed missing CustomerID
Converted InvoiceDate to datetime format
Removed negative quantities (returns)

3️ Feature Engineering

Created customer-level features:
Recency – Days since last purchase
Frequency – Number of purchases
Monetary Value – Total amount spent
Customer Tenure – Active duration
Average Order Value

4️ Risk Score Calculation

A churn risk score is calculated based on:
High Recency → Higher risk
Low Frequency → Higher risk
Low Monetary Value → Higher risk
Final Risk Score formula (weighted scoring approach):
Risk Score =
(Recency Weight × Normalized Recency) +
(Frequency Weight × Inverse Frequency) +
(Monetary Weight × Inverse Monetary)

Customers are categorized as:
High Risk
Medium Risk
Low Risk


Machine Learning Model

Model Used:
Decision Tree Classifier

Why Decision Tree?

Easy to interpret
Handles non-linear relationships
No need for feature scaling
Clear visualization for business explanation

Model Evaluation Metrics:

Accuracy
Precision
Recall
F1 Score
ROC-AUC Score


Dashboard & Visualization

Power BI Dashboard includes:

Total Customers
High-Risk Customers Count
Risk Distribution
Recency vs Frequency Analysis
Customer Segmentation
Risk Score Breakdown
This helps business teams easily identify at-risk customers.


Business Impact

Early identification of potential churn
Improved customer retention strategy
Revenue protection
Better customer engagement planning