# Comparative Analysis of ML Algorithms

## Project Purpose
This project compares three classical machine learning algorithms on two very different data types:
- Tabular data (business records)
- Image data (fruit images)

The goal is to understand how K-Nearest Neighbors, Decision Tree, and Naive Bayes behave across structured numeric features versus flattened pixel features.

## What This Project Does
1. Loads and preprocesses a tabular dataset.
2. Trains and evaluates three models on tabular data.
3. Loads and preprocesses an image dataset.
4. Trains and evaluates the same three models on image data.
5. Compares performance using standard metrics.

## Algorithms Used
- K-Nearest Neighbors (KNN)
- Decision Tree
- Gaussian Naive Bayes

## Evaluation Metrics
For each model, the following metrics are computed:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)

## Workflow Summary
### Part 1: Setup
- Import libraries for data processing, model training, and evaluation.
- Define a reusable evaluation function.

### Part 2: Tabular Pipeline
- Load CSV file.
- Select a subset of meaningful features.
- Encode categorical columns.
- Split into train and test sets.
- Apply feature scaling for fair distance-based modeling.
- Train and evaluate all models.

### Part 3: Image Pipeline
- Read image files from class folders.
- Convert images to grayscale.
- Resize images to a fixed shape.
- Normalize pixel values to the range [0, 1].
- Flatten each image into a 1D vector for classical ML models.
- Encode class labels, split into train and test sets.
- Train and evaluate all models.

### Part 4: Comparison
- Print result tables for tabular and image experiments.
- Summarize key observations.

## Dataset Explanation

### 1) Tabular Dataset
File:
- Superstore Dataset.csv

Type:
- Structured business dataset

Columns used in this project:
- Features: Sales, Quantity, Discount, Profit, Segment, Region
- Target: Ship Mode

Why this dataset is used:
- It represents standard structured data where classical ML algorithms usually perform well.

### 2) Image Dataset
Folder:
- MY_data

Current structure:
- MY_data/train/<class_name>/*.jpeg
- MY_data/test/<class_name>/*.jpeg
- MY_data/predict/*.jpeg

Classes found:
- Apple
- avocado
- Banana
- cherry
- kiwi
- mango
- orange
- pinenapple
- strawberries
- watermelon

Important note:
- In the test folder, class names are not fully consistent in capitalization and spelling (for example apple vs Apple, banana vs Banana, stawberries vs strawberries).
- This project currently trains from the train folder structure and is still valid for comparison experiments.

## Project Structure
- ML_Algorithms_Comparison.ipynb: Main notebook with full step-by-step pipeline.
- run_comparison.py: Script version generated from the notebook.
- Superstore Dataset.csv: Tabular dataset.
- MY_data/: Image dataset root.

## How To Run
### Option A: Notebook
1. Open ML_Algorithms_Comparison.ipynb.
2. Select your Python virtual environment kernel.
3. Run cells from top to bottom.

### Option B: Python Script
Run:
python run_comparison.py

## Requirements
Install main packages:
- pandas
- numpy
- scikit-learn
- pillow
- ipykernel (for notebook usage)

Example install command:
pip install pandas numpy scikit-learn pillow ipykernel

## Expected Output
- Console logs confirming successful training for each model.
- Two comparison tables:
  - Tabular data model results
  - Image data model results
- A short conclusion on why classical ML behaves differently across data types.

## Reusing This In Another Project
Use this as a template by replacing:
1. Tabular CSV file path and selected feature columns.
2. Image dataset folder path and class folders.
3. Optional preprocessing settings:
   - IMAGE_SIZE
   - MAX_IMAGES_PER_CLASS
4. Target column for tabular classification.

Then rerun the same pipeline to compare model behavior on your new datasets.
