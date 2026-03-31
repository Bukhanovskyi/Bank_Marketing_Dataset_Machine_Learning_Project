# Bank Marketing Dataset Machine Learning Project

A full machine learning pipeline for predicting whether a bank client will subscribe to a term deposit, built as a mid-term project for an ML course.

---

## Problem Statement

A Portuguese bank ran phone-based marketing campaigns to sell term deposit subscriptions. The goal is to build a binary classifier that predicts whether a client will subscribe (`yes` / `no`) based on client demographics, campaign data, and macroeconomic indicators.

**Key challenge:** The dataset is heavily imbalanced — only **11.3%** of clients subscribed — making accuracy a misleading metric and requiring careful handling of class imbalance throughout the pipeline.

| Property | Value |
|---|---|
| Rows | 41,188 |
| Features | 21 |
| Target | `y` (yes / no) |
| Positive class | 4,640 (11.3%) |
| Negative class | 36,548 (88.7%) |
| Class imbalance ratio | 7.9 : 1 |

---

## Dataset

**Source:** [Kaggle](https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv)  

**File:** `bank-additional-full.csv`

## About Dataset
### Abstract

The data is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y).

### Information

The data is related to direct marketing campaign direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.
bank-additional-full.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014]
The smallest datasets are provided to test more computationally demanding machine learning algorithms (e.g., SVM). The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).

### Attributes

Input variables:

**bank client data:**

1 - age (numeric)

2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')

3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)

4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')

5 - default: has credit in default? (categorical: 'no','yes','unknown')

6 - housing: has housing loan? (categorical: 'no','yes','unknown')

7 - loan: has personal loan? (categorical: 'no','yes','unknown')

**related with the last contact of the current campaign:**

8 - contact: contact communication type (categorical: 'cellular','telephone')

9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', …, 'nov', 'dec')

10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')

11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

**other attributes:**

12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)

14 - previous: number of contacts performed before this campaign and for this client (numeric)

15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

**social and economic context attributes**

16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)

17 - cons.price.idx: consumer price index - monthly indicator (numeric)

18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)

19 - euribor3m: euribor 3 month rate - daily indicator (numeric)

20 - nr.employed: number of employees - quarterly indicator (numeric)

**Output variable (desired target):**

21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

## Models

| Model | Test ROC-AUC | Test F1 | 
|---|---|---|
| Logistic Regression |0.7932|0.4250| 
| KNN |0.7825|0.1933| 
| Decision Tree |0.8010|0.4603| 
| XGBoost + RandomSearch|0.8071|0.5100| 
| XGBoost + Hyperopt|0.8140|0.5049| 
| LightGBM |0.8141|0.5049|
| LightGBM with 0.54 threshold |0.8141|0.5112|

### Evaluation Strategy
- **Primary metric:** ROC-AUC — threshold-independent, robust to imbalance
- **Secondary metric:** F1 for balance between positive and negative class

### Threshold Tuning

Default threshold of 0.5 is wrong for 11% positive class. Optimal threshold found by sweeping 0.05–0.95 and maximizing F1:

```python
best_threshold = 0.54
F1 improves from 0.5030 (default) to 0.5112 (tuned)
```

---

##  Key Findings

**From EDA:**
- `euribor3m` is the strongest predictor — low interest rates drive subscription
- `contacted_before` shows dramatic effect: 63.8% conversion for recently contacted vs 8.8% for never contacted
- `period` matters hugely: Dec-Mar campaigns convert at 50% vs May at only 6.4%

**From Feature Importance:**
- `euribor3m` dominates all other features
- `contact_cellular` is the second most important feature
- `period_may` acts as a strong negative signal

**From Error Analysis:**
- Model misses ~5% of actual subscribers (False Negatives)
- Ьodel incorrectly classifies ~7% of non-subscribers as subscribers (false positives)
