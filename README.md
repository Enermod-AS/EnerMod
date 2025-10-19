# EnerMod

**EnerMod** is a machine learning project designed to tackle the challenge of **Optimizing Energy Consumption**. Unlike passive monitoring systems, EnerMod implements an **active, intelligent energy management solution** that helps automate energy savings and improve grid stability at the household level.

---

## Project Overview

This repository contains the **first phase** of the EnerMod project, focusing on data acquisition, preprocessing, and preparation for machine learning modeling.

---

## 1. Data Acquisition

The datasets used for model development were sourced from:

- [Kaggle: Electric Power Consumption Data Set](https://www.kaggle.com/uciml/electric-power-consumption-data-set)  
- [UCI Machine Learning Repository: Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

These datasets provide detailed household electricity consumption measurements over time, including active, reactive power, voltage, and sub-metering readings.

---

## 2. Data Preprocessing

The following steps were applied to prepare the data for modeling:

1. **Handling Missing Values**  
   - Missing measurements were identified and imputed or removed.

2. **Data Smoothing**  
   - Exponential smoothing was applied to reduce noise in the signal.

3. **Outlier Detection and Handling**  
   - Outliers were detected using standard deviation thresholds and processed accordingly.

4. **Data Normalization**  
   - Features were scaled to a [0, 1] range for uniformity and model compatibility.

5. **Data Resampling**  
   - Temporal resampling was performed to ensure consistent time intervals (e.g., 1-minute or 1-hour intervals).

6. **Dataset Splitting**  
   - Training Set  
   - Validation Set  
   - Test Set  

These preprocessing steps ensure the dataset is clean, consistent, and ready for machine learning model training and evaluation.

---

## 3. Next Steps

The next phase of the project involves developing and training predictive models for energy consumption optimization, followed by deployment in a household-level intelligent energy management system.

---

## 4. References

- Kaggle dataset: [https://www.kaggle.com/uciml/electric-power-consumption-data-set](https://www.kaggle.com/uciml/electric-power-consumption-data-set)  
- UCI ML Repository: [https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

---

## 5. License

*(Add license info here if applicable, e.g., MIT, GPL, etc.)*

---

