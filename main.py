import os
import sys

# ==========================================
# CUDA / GPU Configuration Setup
# ==========================================
if os.name == 'nt':
    nvidia_bins = []
    for p in sys.path:
        nvidia_path = os.path.join(p, 'nvidia')
        if os.path.exists(nvidia_path):
            for folder in os.listdir(nvidia_path):
                bin_path = os.path.join(nvidia_path, folder, 'bin')
                if os.path.exists(bin_path):
                    nvidia_bins.append(bin_path)
                    try:
                        os.add_dll_directory(bin_path)
                    except AttributeError:
                        pass
    if nvidia_bins:
        os.environ['PATH'] = os.pathsep.join(nvidia_bins) + os.pathsep + os.environ.get('PATH', '')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

# ==========================================
# Reproducibility Setup
# ==========================================
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# CUDA / GPU Configuration
# ==========================================

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Using GPU: {gpus}")
else:
    print("No GPU detected — using CPU")
os.makedirs("results", exist_ok=True)
csv_path = os.path.join("data", "MachineLearningCSV", "MachineLearningCVE", "merged_all_dataset.csv")

if not os.path.exists(csv_path):
    print(f"Error: Dataset not found at {csv_path}")
    exit()

# ==========================================
# Load Dataset
# ==========================================

# Read CSV
try:
    df = pd.read_csv(csv_path, low_memory=False, on_bad_lines='skip')
except Exception as e:
    print(f"Failed to read CSV with default engine: {e}")
    print("Falling back to Python engine...")
    df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')


# ==========================================
# Data Cleaning
# ==========================================

# Remove missing values

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Remove duplicates

df.drop_duplicates(inplace=True)

print('Dataset Shape:', df.shape)

# ==========================================
# Feature Selection
# ==========================================

X = df.drop('Label', axis=1)
y = df['Label']

# ==========================================
# Encode Labels
# ==========================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert to categorical

y_categorical = to_categorical(y_encoded)

# ==========================================
# Train / Validation / Test Split
# ==========================================

# Split BEFORE scaling to prevent data leakage
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

# Stratified validation split from training set (shuffle=True)
y_train_full_classes = np.argmax(y_train_full, axis=1)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42,
    shuffle=True, stratify=y_train_full_classes
)

# ==========================================
# Feature Scaling (fit ONLY on training data)
# ==========================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ==========================================
# Build Model Function
# ==========================================

def build_model(optimizer_choice):

    model = Sequential()

    model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],),
                    kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(128, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(64, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(y_categorical.shape[1], activation='softmax'))

    model.compile(
        optimizer=optimizer_choice,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ==========================================
# Compute Class Weights
# ==========================================

y_train_classes = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_classes),
    y=y_train_classes
)
class_weight_dict = {
    int(k): float(v) for k, v in zip(np.unique(y_train_classes), class_weights)
}

print("\nComputed Class Weights:")
for cls, weight in class_weight_dict.items():
    print(f"Class {cls}: {weight:.4f}")

# ==========================================
# Experiment 1 - Adam
# ==========================================

adam_model = build_model(Adam(learning_rate=0.001, clipnorm=1.0)) # Gradient Clipping (prevents exploding gradients)

early_stop = EarlyStopping(
    monitor='val_loss', patience=5,
    restore_best_weights=True, verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=2,
    min_lr=1e-6, verbose=1
)

history_adam = adam_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=256,
    verbose=1,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr]
)

# Evaluation
adam_loss, adam_accuracy = adam_model.evaluate(X_test, y_test)
adam_preds = np.argmax(adam_model.predict(X_test), axis=1)
true_classes = np.argmax(y_test, axis=1)
adam_f1 = f1_score(true_classes, adam_preds, average='weighted')
adam_precision = precision_score(true_classes, adam_preds, average='weighted')
adam_recall = recall_score(true_classes, adam_preds, average='weighted')

print('\nAdam Results')
print(f'  Accuracy:  {adam_accuracy:.4f}')
print(f'  F1 Score:  {adam_f1:.4f}')
print(f'  Precision: {adam_precision:.4f}')
print(f'  Recall:    {adam_recall:.4f}')
print(f'  Loss:      {adam_loss:.4f}')


# ==========================================
# Experiment 2 - SGD (with momentum)
# ==========================================

sgd_model = build_model(SGD(learning_rate=0.001, momentum=0.9, clipnorm=1.0))

early_stop = EarlyStopping(
    monitor='val_loss', patience=5,
    restore_best_weights=True, verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=2,
    min_lr=1e-6, verbose=1
)

history_sgd = sgd_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=256,
    verbose=1,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr]
)

# Evaluation
sgd_loss, sgd_accuracy = sgd_model.evaluate(X_test, y_test)
sgd_preds = np.argmax(sgd_model.predict(X_test), axis=1)
sgd_f1 = f1_score(true_classes, sgd_preds, average='weighted')
sgd_precision = precision_score(true_classes, sgd_preds, average='weighted')
sgd_recall = recall_score(true_classes, sgd_preds, average='weighted')

print('\nSGD Results')
print(f'  Accuracy:  {sgd_accuracy:.4f}')
print(f'  F1 Score:  {sgd_f1:.4f}')
print(f'  Precision: {sgd_precision:.4f}')
print(f'  Recall:    {sgd_recall:.4f}')
print(f'  Loss:      {sgd_loss:.4f}')

# ==========================================
# Results Table
# ==========================================

results = pd.DataFrame({
    'Model': ['Adam', 'SGD'],
    'Accuracy': [adam_accuracy, sgd_accuracy],
    'F1 Score': [adam_f1, sgd_f1],
    'Precision': [adam_precision, sgd_precision],
    'Recall': [adam_recall, sgd_recall],
    'Loss': [adam_loss, sgd_loss]
})

print('\nResults Comparison')
print(results.to_string(index=False))

plt.figure(figsize=(10, 5))
x = np.arange(len(results['Model']))
width = 0.2
plt.bar(x - width, results['Accuracy'], width, label='Accuracy')
plt.bar(x, results['F1 Score'], width, label='F1 Score')
plt.bar(x + width, results['Precision'], width, label='Precision')
plt.xticks(x, results['Model'])
plt.title('Results Comparison')
plt.ylabel('Score')
plt.legend()
plt.savefig('results/results_comparison.png', dpi=150, bbox_inches='tight')
print('Saved: results/results_comparison.png')
plt.close()

# ==========================================
# Select Best Model
# ==========================================

if adam_accuracy >= sgd_accuracy:
    print('\nSelecting Adam model as best based on accuracy.')
    best_model = adam_model
    best_name = 'Adam'
    best_history = history_adam
else:
    print('\nSelecting SGD model as best based on accuracy.')
    best_model = sgd_model
    best_name = 'SGD'
    best_history = history_sgd

# ==========================================
# Visualization - Accuracy
# ==========================================


plt.figure(figsize=(10,5))

plt.plot(best_history.history['accuracy'], label=f'Train Accuracy - {best_name}')
plt.plot(best_history.history['val_accuracy'], label=f'Validation Accuracy - {best_name}')

plt.title(f'{best_name} Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.savefig(f'results/{best_name.lower()}_accuracy.png', dpi=150, bbox_inches='tight')
print(f'Saved: results/{best_name.lower()}_accuracy.png')
plt.close()

# ==========================================
# Visualization - Loss
# ==========================================

plt.figure(figsize=(10,5))

plt.plot(best_history.history['loss'], label=f'Train Loss - {best_name}')
plt.plot(best_history.history['val_loss'], label=f'Validation Loss - {best_name}')

plt.title(f'{best_name} Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.savefig(f'results/{best_name.lower()}_loss.png', dpi=150, bbox_inches='tight')
print(f'Saved: results/{best_name.lower()}_loss.png')
plt.close()

# ==========================================
# Predictions
# ==========================================

predictions = best_model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

print('\nClassification Report')
print(classification_report(true_classes, predicted_classes,
      target_names=label_encoder.classes_))

# ==========================================
# Confusion Matrix Heatmap
# ==========================================

cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix — {best_name} Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
print('Saved: results/confusion_matrix.png')
plt.close()
