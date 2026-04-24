# Classical ML Comparison (College Assignment)

This project is a college assignment created to understand how different classical ML models behave on different data types.

## Objective
Compare these 3 models on tabular and image data:
- K-Nearest Neighbors
- Decision Tree
- Naive Bayes

## Datasets
- Tabular: `Superstore Dataset.csv`
- Image: fruit dataset in `MY_data/train` (Apple, avocado, Banana, cherry, kiwi, mango, orange, pinenapple, strawberries, watermelon)

## Results Shared

### Tabular Data Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| K-Nearest Neighbors | 0.474237 | 0.442097 | 0.474237 | 0.456354 |
| Decision Tree | 0.411706 | 0.432905 | 0.411706 | 0.421596 |
| Naive Bayes | 0.615808 | 0.420299 | 0.615808 | 0.473205 |

### Image Data Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| K-Nearest Neighbors | 0.165000 | 0.170922 | 0.165000 | 0.150764 |
| Decision Tree | 0.168333 | 0.170522 | 0.168333 | 0.168783 |
| Naive Bayes | 0.190000 | 0.226502 | 0.190000 | 0.163465 |

## Run
- Notebook: `ML_Algorithms_Comparison.ipynb`
- Script: `run_comparison.py`
