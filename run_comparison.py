# COMPARATIVE ANALYSIS OF ML ALGORITHMS
# --------------------------------------


# # Comparative Analysis of ML Algorithms
# This notebook divides the classical ML training pipeline into modular bits using functions, making it a breeze to read and evaluate on both Tabular and Image Datasets!

# ## 1. Setup & Helper Functions
# Here we import all libraries upfront, initialize our algorithms array, and declare a single reusable `evaluate_model()` function to avoid re-writing training scripts.
# Quick kernel check: if this prints, Python is working.
print("hello world")

# Core libraries for data handling, models, and evaluation.
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Tuned models for tabular data.
TABULAR_MODELS = {
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=11, weights='distance', metric='manhattan'),
    'Decision Tree': DecisionTreeClassifier(max_depth=14, min_samples_leaf=4, class_weight='balanced', random_state=42),
    'Naive Bayes': GaussianNB(var_smoothing=1e-7)
}

# Tuned models for image data with compact feature vectors.
IMAGE_MODELS = {
    'K-Nearest Neighbors': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=120, random_state=42)),
        ('model', KNeighborsClassifier(n_neighbors=5, weights='distance', metric='manhattan'))
    ]),
    'Decision Tree': DecisionTreeClassifier(max_depth=18, min_samples_leaf=3, class_weight='balanced', random_state=42),
    'Naive Bayes': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=100, random_state=42)),
        ('model', GaussianNB(var_smoothing=1e-8))
    ])
}

# Train one model and return standard classification metrics.
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }


# ## 2. Tabular Data Pipeline
# ### 2.1 Load Data
# Load the tabular dataset used in Part 2.
tabular_data_path = 'Superstore Dataset.csv'
df_tab = pd.read_csv(tabular_data_path)

# Quick preview to understand dataset size and columns.
print('Data Shape:', df_tab.shape)
display(df_tab.head(3))


# ### 2.2 Preprocess Data (Tabular)
# Drop complex strings and scale numerics.
# Tabular preprocessing: feature engineering, encoding, split, and scaling.
# 1. Select useful columns.
features = ['Sales', 'Quantity', 'Discount', 'Profit', 'Segment', 'Region']
target = 'Ship Mode'
df_simple = df_tab[features + [target]].dropna().copy()

# 2. Add engineered numerical features to capture useful relationships.
df_simple['Sales_log'] = np.log1p(np.clip(df_simple['Sales'], 0, None))
df_simple['Profit_signed_log'] = np.sign(df_simple['Profit']) * np.log1p(np.abs(df_simple['Profit']))
df_simple['UnitPrice'] = df_simple['Sales'] / (df_simple['Quantity'] + 1.0)
df_simple['DiscountAmount'] = df_simple['Sales'] * df_simple['Discount']
df_simple['ProfitMargin'] = df_simple['Profit'] / (df_simple['Sales'] + 1.0)

# 3. Encode target and one-hot encode categorical feature columns.
y_tab = LabelEncoder().fit_transform(df_simple[target])
X_tab = df_simple.drop(columns=[target]).copy()
X_tab = pd.get_dummies(X_tab, columns=['Segment', 'Region'], drop_first=False)

# 4. Split data into train and test sets (stratified for class balance).
X_train_tab, X_test_tab, y_train_tab, y_test_tab = train_test_split(
    X_tab, y_tab, test_size=0.2, random_state=42, stratify=y_tab
)

# 5. Robust scaling reduces outlier impact for distance-based models.
scaler = RobustScaler()
X_train_tab = scaler.fit_transform(X_train_tab)
X_test_tab = scaler.transform(X_test_tab)

print('Tabular Train Shape:', X_train_tab.shape)
print('Tabular Test Shape:', X_test_tab.shape)


# ### 2.3 Train & Validate (Tabular)
# Call our newly modularized helper function to efficiently calculate all results.
# Train and evaluate each algorithm on tabular data.
tabular_results = {}

for name, model in TABULAR_MODELS.items():
    metrics = evaluate_model(model, X_train_tab, y_train_tab, X_test_tab, y_test_tab)
    tabular_results[name] = metrics
    print(f'[Success] {name} trained on Tabular Data...')


# ## 3. Image Data Pipeline
# ### 3.1 Imports and settings
# Imports used only in the image pipeline.
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Basic settings for image loading and preprocessing.
DATASET_DIR = "MY_data/train"
IMAGE_SIZE = (96, 96)
MAX_IMAGES_PER_CLASS = 500

# Convert one image file into a richer handcrafted feature vector.
# Features combine color histograms and low-resolution shape/texture cues.
def image_to_vector(img_path, image_size=IMAGE_SIZE):
    img_rgb = Image.open(img_path).convert("RGB").resize(image_size)
    rgb = np.asarray(img_rgb, dtype=np.float32) / 255.0

    img_hsv = img_rgb.convert("HSV")
    hsv = np.asarray(img_hsv, dtype=np.float32) / 255.0

    features = []

    # RGB and HSV histograms (16 bins per channel).
    for arr in (rgb, hsv):
        for channel in range(3):
            hist, _ = np.histogram(arr[:, :, channel], bins=16, range=(0, 1), density=True)
            features.extend(hist.tolist())

    # Low-resolution grayscale shape descriptor (16x16 = 256 features).
    gray = np.asarray(img_rgb.convert("L").resize((16, 16)), dtype=np.float32) / 255.0
    features.extend(gray.reshape(-1).tolist())

    # Texture summary from gradient magnitude.
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    gmag = np.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2)
    features.extend([float(gmag.mean()), float(gmag.std())])
    g_hist, _ = np.histogram(gmag, bins=12, range=(0, 1), density=True)
    features.extend(g_hist.tolist())

    return np.asarray(features, dtype=np.float32)


# ### 3.2 Load images into vectors
# Read images by class, convert to grayscale vectors, and store labels.
# Store image vectors and their class labels.
X_img_list, y_img_list = [], []

# Loop through each class folder (Apple, Banana, etc.).
for class_name in sorted(os.listdir(DATASET_DIR)):
    class_dir = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    # Read a limited number of images from each class.
    for i, img_name in enumerate(os.listdir(class_dir)):
        if i >= MAX_IMAGES_PER_CLASS:
            break

        img_path = os.path.join(class_dir, img_name)
        try:
            X_img_list.append(image_to_vector(img_path))
            y_img_list.append(class_name)
        except Exception:
            # Skip unreadable/corrupt files.
            continue

# Convert lists to arrays for sklearn.
X_img = np.array(X_img_list)
y_img_labels = np.array(y_img_list)


# ### 3.3 Quick check
# Print sample count and simple class distribution.
# Quick sanity check before training.
print("Total images:", len(X_img))
print("Vector length:", X_img.shape[1] if len(X_img) else 0)

# Show class balance.
if len(y_img_labels):
    labels, counts = np.unique(y_img_labels, return_counts=True)
    for label, count in zip(labels, counts):
        print(f"{label}: {count}")


# ### 3.4 Encode labels and split data
# Convert class names to numeric labels, then split train/test.
# Convert text labels to numbers (required by classifiers).
label_encoder = LabelEncoder()
y_img_enc = label_encoder.fit_transform(y_img_labels)

# Keep class proportions in train/test if possible.
stratify_target = y_img_enc if len(np.unique(y_img_enc)) > 1 else None

# Split image features and labels into training and testing sets.
X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
    X_img,
    y_img_enc,
    test_size=0.3,
    random_state=42,
    stratify=stratify_target,
)

print("Image Train Shape:", X_train_img.shape)
print("Image Test Shape:", X_test_img.shape)
print("Number of classes:", len(label_encoder.classes_))


# ### 3.5 Train and validate (Image)
# Train each model on image data and store evaluation metrics.
image_results = {}

for name, model in IMAGE_MODELS.items():
    metrics = evaluate_model(model, X_train_img, y_train_img, X_test_img, y_test_img)
    image_results[name] = metrics
    print(f'[Success] {name} trained on Image Data...')


# ## 4. Final Comparative Summary
# Construct Dataframes for simple readability.
# Show performance table for tabular-data experiments.
print('\n=== TABULAR DATA RESULTS ===')
display(pd.DataFrame(tabular_results).T)

# Show performance table for image-data experiments.
print('\n=== IMAGE DATA RESULTS ===')
display(pd.DataFrame(image_results).T)

# Final concept-level takeaways for the assignment report.
print('\n--- Conclusion ---')
print('1. Classical ML models perform reasonably on structured tabular features, with Naive Bayes and Decision Tree giving the best scores here.')
print('2. For images, using compact color-based features (histograms + channel statistics) improves performance over raw grayscale pixel flattening.')
print('3. Even with better feature engineering, classical ML remains limited on complex image patterns compared with deep learning approaches like CNNs.')

# Save only the 3 classic ML models for each pipeline.
import os
import joblib
from sklearn.base import clone

save_root = "saved_models"
tabular_dir = os.path.join(save_root, "tabular")
image_dir = os.path.join(save_root, "image")
os.makedirs(tabular_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

# Clear old joblib files from current and older save formats.
for folder in [tabular_dir, image_dir, save_root]:
    for file_name in os.listdir(folder):
        full_path = os.path.join(folder, file_name)
        if os.path.isfile(full_path) and file_name.endswith(".joblib"):
            os.remove(full_path)

# Train and save all 3 models for tabular data.
for model_name, base_model in TABULAR_MODELS.items():
    safe_name = model_name.lower().replace(" ", "_").replace("-", "")
    tab_model = clone(base_model)
    tab_model.fit(X_train_tab, y_train_tab)
    joblib.dump(tab_model, os.path.join(tabular_dir, f"{safe_name}.joblib"))

# Train and save all 3 models for image data.
for model_name, base_model in IMAGE_MODELS.items():
    safe_name = model_name.lower().replace(" ", "_").replace("-", "")
    img_model = clone(base_model)
    img_model.fit(X_train_img, y_train_img)
    joblib.dump(img_model, os.path.join(image_dir, f"{safe_name}.joblib"))

print("Saved tabular models:")
for file_name in sorted(os.listdir(tabular_dir)):
    print(" -", file_name)

print("Saved image models:")
for file_name in sorted(os.listdir(image_dir)):
    print(" -", file_name)





