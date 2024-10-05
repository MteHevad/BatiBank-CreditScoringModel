# BatiBank-CreditScoringModel
---
## Project Overview
This repository contains the implementation of a **Credit Scoring Model** for Bati Bank as part of the **10 Academy AI Mastery Week 6 Challenge**. The model is designed to predict customer creditworthiness for a buy-now-pay-later (BNPL) service in partnership with an eCommerce platform. The goal is to categorize users into **high risk** (likely to default) or **low risk** (likely to repay), and predict loan amounts and durations.
Bati Bank seeks to provide a credit-scoring solution for its customers applying for credit on the eCommerce platform. This project involves:
- **Understanding Credit Risk**: Researching credit risk and identifying factors to classify users.
- **Exploratory Data Analysis (EDA)**: Analyzing the dataset to uncover patterns, distributions, and correlations.
- **Feature Engineering**: Creating new features from raw data to improve model accuracy.
- **Model Training**: Developing and evaluating machine learning models to predict credit risk.
- **Model Deployment**: Serving the model through an API for real-time predictions.
---
## Data
The dataset consists of customer transaction data from the eCommerce platform, with the following key features:
- `TransactionId`: Unique identifier for each transaction.
- `CustomerId`: Unique identifier for each customer.
- `Amount`: Value of the transaction.
- `ProductCategory`: Category of the purchased product.
- `ChannelId`: Platform used for transactions (Web, Android, iOS, etc.).
- `FraudResult`: Indicator of whether the transaction was fraudulent (1 = yes, 0 = no).
- Several other transactional and customer-related features are also included.
