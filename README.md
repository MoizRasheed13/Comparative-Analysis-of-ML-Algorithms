# Classical ML Comparison On Tabular And Image Data

## Assignment Context
This repository is a college assignment submission created to understand how different classical machine learning models behave on different data types.

The same three algorithms are tested on:
- Tabular business data
- Image-based fruit classification data

Target models:
- K-Nearest Neighbors (KNN)
- Decision Tree
- Naive Bayes

## Assignment Objective
1. Build complete preprocessing pipelines for tabular and image data.
2. Train and evaluate KNN, Decision Tree, and Naive Bayes on both datasets.
3. Compare the behavior of classical ML across structured and visual inputs.
4. Save trained models and expose them through a simple Flask web interface.

## Project Files
- `ML_Algorithms_Comparison.ipynb`: Main assignment notebook (full workflow, results, and model saving).
- `flask_app/`: Web app for inference from UI.
- `saved_models/tabular/`: Saved tabular models.
- `saved_models/image/`: Saved image models.
- `Superstore Dataset.csv`: Tabular dataset.
- `MY_data/train/`: Fruit image dataset.

## Dataset Details

### 1) Tabular Dataset
- File: `Superstore Dataset.csv`
- Task: Predict `Ship Mode`
- Base input columns used:
  - Sales, Quantity, Discount, Profit, Segment, Region

### 2) Image Dataset
- Folder: `MY_data/train`
- Task: Fruit image classification
- Classes:
  - Apple, avocado, Banana, cherry, kiwi, mango, orange, pinenapple, strawberries, watermelon

## Methodology

### Tabular Preprocessing
1. Remove missing rows from selected columns.
2. Feature engineering:
	- `Sales_log`
	- `Profit_signed_log`
	- `UnitPrice`
	- `DiscountAmount`
	- `ProfitMargin`
3. One-hot encoding for `Segment` and `Region`.
4. Stratified train/test split.
5. Robust scaling (`RobustScaler`) before model training.

### Image Preprocessing
1. Load and resize RGB images.
2. Build compact handcrafted feature vectors using:
	- RGB histograms
	- HSV histograms
	- Low-resolution grayscale shape descriptor
	- Gradient-based texture summary
3. Encode class labels.
4. Stratified train/test split.

## Model Configuration

### Tabular Models
- KNN: `n_neighbors=11`, `weights='distance'`, `metric='manhattan'`
- Decision Tree: `max_depth=14`, `min_samples_leaf=4`, `class_weight='balanced'`
- Naive Bayes: `var_smoothing=1e-7`

### Image Models
- KNN Pipeline: `StandardScaler + PCA(120) + KNN`
- Decision Tree: `max_depth=18`, `min_samples_leaf=3`, `class_weight='balanced'`
- Naive Bayes Pipeline: `StandardScaler + PCA(100) + GaussianNB`

## Evaluation Metrics
The assignment reports:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)

## Final Results (Latest Notebook Run)

### Tabular Data Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| K-Nearest Neighbors | 0.527764 | 0.422858 | 0.527764 | 0.455511 |
| Decision Tree | 0.261631 | 0.426041 | 0.261631 | 0.295272 |
| Naive Bayes | 0.558779 | 0.400417 | 0.558779 | 0.450950 |

### Image Data Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| K-Nearest Neighbors | 0.328509 | 0.332056 | 0.328509 | 0.322713 |
| Decision Tree | 0.322721 | 0.329484 | 0.322721 | 0.324710 |
| Naive Bayes | 0.269175 | 0.316730 | 0.269175 | 0.265701 |

## Interpretation
1. Tabular data gives stronger performance with classical ML due to structured numerical patterns.
2. Image results improve after feature engineering but remain lower than tabular.
3. This demonstrates that classical ML depends heavily on input representation quality, especially for images.

## How To Run

### Notebook (Training + Saving Models)
1. Open `ML_Algorithms_Comparison.ipynb`.
2. Run all cells in order.
3. Models are saved to:
	- `saved_models/tabular/`
	- `saved_models/image/`

### Flask App (Inference UI)
1. Go to `flask_app/`
2. Install dependencies:
	- `pip install -r requirements.txt`
3. Run server:
	- `python app.py`
4. Open:
	- `http://127.0.0.1:5000`

UI features:
- Tabular and Image tabs
- Model switching
- Input guidance text and sample ranges
- Prediction confidence visualization

## Conclusion
This assignment successfully demonstrates how the same classical ML models behave differently on tabular vs image data, why preprocessing quality matters, and how trained models can be deployed in a simple interactive Flask application.
