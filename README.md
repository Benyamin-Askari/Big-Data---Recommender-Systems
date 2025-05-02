# Big-Data---Recommender-Systems
this is one of the assignments for the Big data module for my MSc in IoT Data Science

%md
# **Task 2** - Steam 200k Dataset

%md
# Table of Contents

## 1. Load Data
- 1.1 Header Inspection
- 1.2 Load Dataset with Column Names

## 2. Exploratory Data Analysis (EDA)
- 2.1 Data Summary
- 2.2 Missing Values
- 2.3 Duplicate Values
  - 2.3.1 Row-based Duplicates
  - 2.3.2 Game-level Duplicates per User
  - 2.3.3 Action-level Duplicates
  - 2.3.4 Sum of Duplicates
  - 2.3.5 Action Type Validation
- 2.4 Detecting Unusual Values
  - 2.4.1 Invalid User IDs
  - 2.4.2 Invalid Game Names
  - 2.4.3 Invalid Value Combinations

## 3. Game-User Behavior Patterns
- 3.1 Import and Filtering Setup
- 3.2 Game-Level Metrics
  - 3.2.1 Total Purchases per Game
  - 3.2.2 Total Play Instances per Game
  - 3.2.3 Total Play Hours per Game
  - 3.2.4 Purchase Frequency Segments
  - 3.2.5 Purchased but Not Played
  - 3.2.6 Purchased and Played
  - 3.2.7 Played but Not Purchased
  - 3.2.8 Average Playtime per User
  - 3.2.9 Maximum Playtime per User
  - 3.2.10 Minimum Playtime per User
  - 3.2.11 Final Game-Level Summary
- 3.3 User-Level Interaction Distribution
- 3.4 Sparsity
- 3.5 Correlation Checks
  - 3.5.1 Purchase vs Play Instances
  - 3.5.2 Purchase vs Play Hours
  - 3.5.3 Play Instances vs Play Hours
  - 3.5.4 User Purchase vs Play Hours
  - 3.5.5 User Purchase vs Play Instances
  - 3.5.6 User Play Hours vs Play Instances
- 3.6 Top Games and Users by Engagement Metrics

## 4. Recommender System – ALS
- 4.1 ALS on Full Dataset
  - 4.1.1 Indexing and Log Transformation
  - 4.1.2 Train-Test Split
  - 4.1.3 MLflow and Evaluator Setup
  - 4.1.4 Hyperparameter Tuning and MLflow Logging
  - 4.1.5 Load Best Model & Predict
  - 4.1.6 MAE and Residual Analysis
  - 4.1.7 Actual vs Predicted Plot
  - 4.1.8 Per-User RMSE Distribution
  - 4.1.9 Game Recommendation Using ALS
- 4.2 ALS on Filtered Dataset
  - 4.2.1 User Filtering
  - 4.2.2 Filtering Dataset and Preview
  - 4.2.3 Filtered Train-Test Split
  - 4.2.4 Create Filtered MLflow Experiment
  - 4.2.5 Train Filtered ALS Model with MLflow
  - 4.2.6 Load Best Filtered Model and Evaluate
  - 4.2.7 Actual vs Predicted Plot (Filtered)
  - 4.2.8 Per-User RMSE (Filtered)
  - 4.2.9 Display Recommendations
- 4.3 Model Comparison
