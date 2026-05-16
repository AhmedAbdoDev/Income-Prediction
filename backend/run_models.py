import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data
from analysis import run_full_analysis, corr_between_each_col_and_income
import xgboost as xgb
from xgboost import XGBClassifier
from collections import Counter

# 1. LOAD DATA
raw_train_data = pd.read_csv('./dataset/train_data.csv')
raw_test_data  = pd.read_csv('./dataset/test_data.csv')

# 2. VISUALIZATION
run_full_analysis(raw_train_data)

# 3. PREPROCESS
cleaned_train_data = preprocess_data(raw_train_data)
cleaned_test_data  = preprocess_data(raw_test_data)

# 4. ENCODING — convert text columns to numbers
categorical_columns = cleaned_train_data.select_dtypes(include=['object', 'string']).columns.drop('Income')
for column in categorical_columns:
  label_encoder = LabelEncoder()
  cleaned_train_data[column] = label_encoder.fit_transform(cleaned_train_data[column])
  most_common = cleaned_train_data[column].mode()[0]
  cleaned_test_data[column] = cleaned_test_data[column].map(lambda value: value if value in label_encoder.classes_ else most_common)
  cleaned_test_data[column] = label_encoder.transform(cleaned_test_data[column])
# encode target: <=50K → 0 | >50K → 1
income_label_map = {'<=50K': 0, '>50K': 1}
cleaned_train_data['Income'] = cleaned_train_data['Income'].map(income_label_map)
cleaned_test_data['Income']  = cleaned_test_data['Income'].map(income_label_map)

correlation_matrix = cleaned_train_data.corr(numeric_only=True)
corr_between_each_col_and_income(correlation_matrix)

# ════════════════════════════════════════════════════════════
# 5. SPLIT TRAIN DATA → TRAIN (80%) + VALIDATION (20%)
#   Features -> X data to train on            labels -> Y data to predict 

#   train_features / train_labels   → model learns from this
#   val_features   / val_labels     → used to compare & tune models
#   test_features  / test_labels    → touched once at the very end
# ════════════════════════════════════════════════════════════
all_train_features = cleaned_train_data.drop('Income', axis=1)
all_train_labels   = cleaned_train_data['Income']

train_features, val_features, train_labels, val_labels = train_test_split(
  all_train_features,
  all_train_labels,
  test_size    = 0.2,              # 20% → validation
  random_state = 42,               # same split every run
  stratify     = all_train_labels  # keep 75/25 income ratio in both splits
)

test_features = cleaned_test_data.drop('Income', axis=1)
test_labels   = cleaned_test_data['Income']

# ════════════════════════════════════════════════════════════
# 6. SCALING — bring all features to the same range
#    fit_transform on train → transform (only) on val and test
# ════════════════════════════════════════════════════════════
feature_scaler = StandardScaler()
scaled_train_features = feature_scaler.fit_transform(train_features)
scaled_val_features   = feature_scaler.transform(val_features)
scaled_test_features  = feature_scaler.transform(test_features)

# Robust scaling for KNN only
knn_scaler = RobustScaler()
scaled_train_features_knn = knn_scaler.fit_transform(train_features)
scaled_val_features_knn   = knn_scaler.transform(val_features)
scaled_test_features_knn  = knn_scaler.transform(test_features)

# ════════════════════════════════════════════════════════════
# 7. DEFINE MODELS
#    Format → "Name": (model, train_data, validation_data)
#    Scale-sensitive models  → use scaled data
#    Tree-based models       → use raw data (scale doesn't affect them)
# ════════════════════════════════════════════════════════════
counter = Counter(train_labels)
computed_ratio = counter[0] / counter[1]
model_definitions = {
  # ── Logistic Regression — hyperparameter: C ────────────
  # "Logistic Regression (C=0.01)": (LogisticRegression(C=0.01, class_weight='balanced'), scaled_train_features, scaled_val_features),
  "Logistic Regression (C=1.0)": (LogisticRegression(C=1.0, class_weight='balanced'), scaled_train_features, scaled_val_features),
  # ── SVM — hyperparameter: C ────────────────────────────
  # "SVM (C=0.01)": (SVC(kernel='rbf', C=0.01, class_weight='balanced'), scaled_train_features, scaled_val_features),
  "SVM (C=1.0)": (SVC(kernel='rbf', C=1.0, class_weight='balanced'), scaled_train_features, scaled_val_features),
  # ── Decision Tree — hyperparameter: max_depth ──────────
  "Decision Tree (Depth=10)": (DecisionTreeClassifier(max_depth=10, class_weight='balanced'), train_features, val_features),
  # "Decision Tree (Depth=5)": (DecisionTreeClassifier(max_depth=5, class_weight='balanced'), train_features, val_features),
  # "Decision Tree (Depth=None)": (DecisionTreeClassifier(max_depth=None, class_weight='balanced'), train_features, val_features),
  # ── KNN — hyperparameter: n_neighbors ────
  "KNN (K=21)": (KNeighborsClassifier(n_neighbors=21, weights='uniform', metric='minkowski', p=1), scaled_train_features_knn, scaled_val_features_knn),
  #"KNN (K=5)" : (KNeighborsClassifier(n_neighbors=5),scaled_train_features_knn,scaled_val_features_knn),
  # ── Random Forest (Bonus Model) ────────────────────────
  "Random Forest": (RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=2, random_state=42, class_weight='balanced'), train_features, val_features),
  # ── XGBoost (Bonus Model) ────────────────────────
  "XGBoost": (XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.7, colsample_bytree=0.7, min_child_weight=5, random_state=42, scale_pos_weight=computed_ratio), train_features, val_features),
}
def evaluate_model(y_true, y_pred):
  return {
    "accuracy":  accuracy_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred, zero_division=0),
    "recall":    recall_score(y_true, y_pred, zero_division=0),
    "f1":        f1_score(y_true, y_pred, zero_division=0)
  }

# ════════════════════════════════════════════════════════════
# 8. PHASE 1 — TRAIN & EVALUATE ON VALIDATION
#    All model comparison and tuning decisions happen here.
#    Test data is NOT used in this phase.
# ════════════════════════════════════════════════════════════
print("=" * 70)
print("  VALIDATION RESULTS  ")
print("=" * 70)

validation_scores = {}
output_folder = './models_analytics/'
if not os.path.exists(output_folder):
  os.makedirs(output_folder)

for model_name, (model, training_data, validation_data) in model_definitions.items():
  model.fit(training_data, train_labels)
  validation_predictions = model.predict(validation_data)
  metrics = evaluate_model(val_labels, validation_predictions)
  validation_scores[model_name] = {"model": model, **metrics}
  print(f"{model_name:30s} | " f"Acc:{metrics['accuracy']:.4f} " f"Prec:{metrics['precision']:.4f} " f"Rec:{metrics['recall']:.4f} " f"F1:{metrics['f1']:.4f}")
  safe_file_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
  cm = confusion_matrix(val_labels, validation_predictions)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['<=50K', '>50K'])
  disp.plot(cmap='Blues', values_format='d')
  plt.title(f'Confusion Matrix: {model_name}')
  plt.savefig(f"{output_folder}{safe_file_name}_CM.png")
  plt.close('all') 
  is_tree_based = any(x in model_name for x in ["Tree", "Random Forest"]) #, "XGBoost"
  if is_tree_based:
    plt.figure(figsize=(20, 10))
    try:
      if "Decision Tree" in model_name:
        plot_tree(model, feature_names=all_train_features.columns.tolist(), class_names=['<=50K', '>50K'], filled=True, rounded=True, max_depth=3)
        plt.title(f"Tree Structure: {model_name}")
      elif "Random Forest" in model_name:
        plot_tree(model.estimators_[0], feature_names=all_train_features.columns.tolist(), class_names=['<=50K', '>50K'], filled=True, rounded=True, max_depth=3)
        plt.title(f"Random Forest (First Tree): {model_name}")
      # elif "XGBoost" in model_name:
      #   xgb.plot_tree(model, tree_idx=0, rankdir='LR')
      #   plt.title("XGBoost Tree Structure")
      plt.savefig(f"{output_folder}{safe_file_name}_Tree.png", dpi=300, bbox_inches='tight')
    except Exception as e:
      print(f"Could not plot tree for {model_name}: {e}")
    plt.close('all')
best_model_name = max(validation_scores, key=lambda x: validation_scores[x]["f1"])
best_model = validation_scores[best_model_name]["model"]

print(f"\nBest Validation Model: {best_model_name}")

# ════════════════════════════════════════════════════════════
# 9. PHASE 2 — FINAL EVALUATION ON TEST DATA
#    Test data is touched exactly once here.
#    No further tuning is allowed after this step.
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FINAL TEST RESULTS  (official scores)")
print("=" * 70)
final_test_scores = {}
SCALE_SENSITIVE_MODELS = ["Logistic", "SVM", "KNN"]
for model_name, (model, _, _) in model_definitions.items():
  if "KNN" in model_name:
    final_test_data = scaled_test_features_knn
  elif any(keyword in model_name for keyword in ["Logistic", "SVM"]):
    final_test_data = scaled_test_features
  else:
    final_test_data = test_features
  final_predictions = model.predict(final_test_data)
  metrics = evaluate_model(test_labels, final_predictions)
  final_test_scores[model_name] = metrics
  print(f"{model_name:30s} | " f"Acc:{metrics['accuracy']:.4f} " f"Prec:{metrics['precision']:.4f} " f"Rec:{metrics['recall']:.4f} " f"F1:{metrics['f1']:.4f}")

# ════════════════════════════════════════════════════════════
# 10. BEST MODEL SUMMARY
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BEST MODEL SUMMARY")
print("=" * 70)

best_test_metrics = final_test_scores[best_model_name]

print(f"Model: {best_model_name}")
print(f"Validation F1: {validation_scores[best_model_name]['f1']:.4f}")
print(f"Test F1: {best_test_metrics['f1']:.4f}")
print(f"Generalization Gap: {abs(validation_scores[best_model_name]['f1'] - best_test_metrics['f1']):.4f}")

model_dir = './models/'
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
joblib.dump(best_model, f'{model_dir}best_model.pkl')
if "KNN" in best_model_name:
  joblib.dump(knn_scaler, f'{model_dir}scaler.pkl')
else:
  joblib.dump(feature_scaler, f'{model_dir}scaler.pkl')
feature_columns = list(train_features.columns)
joblib.dump(feature_columns, f'{model_dir}feature_columns.pkl')
print("Model saved!")

model_metadata = {
  "model_name": best_model_name,
  "f1": round(final_test_scores[best_model_name]["f1"] * 100, 2),
  "recall": round(final_test_scores[best_model_name]["recall"] * 100, 2),
  "accuracy": round(final_test_scores[best_model_name]["accuracy"] * 100, 2),
  "precision": round(final_test_scores[best_model_name]["precision"] * 100, 2),
}
joblib.dump(model_metadata, f'{model_dir}model_metadata.pkl')
print("Metadata saved!")