import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import cv2
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Set random seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def dilate_digit(img_np, max_kernel=2):
    """
    Apply morphological dilation to a grayscale image (0–255).
    """
    k_size = np.random.randint(1, max_kernel + 1)
    kernel = np.ones((k_size, k_size), np.uint8)
    dilated = cv2.dilate(img_np, kernel, iterations=1)
    return np.clip(dilated, 0, 255).astype(np.uint8)

def erode_digit(img_np, max_kernel=2):
    """
    Apply morphological erosion to a grayscale image (0–255).
    This will thin strokes / open loops.
    """
    k_size = np.random.randint(1, max_kernel + 1)
    kernel = np.ones((k_size, k_size), np.uint8)
    eroded = cv2.erode(img_np, kernel, iterations=1)
    return np.clip(eroded, 0, 255).astype(np.uint8)


class AugmentedDataGenerator(keras.utils.Sequence):
    """Custom data generator that applies targeted augmentation to digit 4s and 9s."""
    
    def __init__(self, x_data, y_data, batch_size=128,
                 prob_dilate_4=0.12, prob_erode_9=0.12, shuffle=True):
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.prob_dilate_4 = prob_dilate_4
        self.prob_erode_9 = prob_erode_9
        self.shuffle = shuffle
        self.indices = np.arange(len(self.x_data))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.x_data) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.zeros((len(batch_indices), 28, 28, 1), dtype=np.float32)
        batch_y = np.zeros((len(batch_indices),), dtype=np.int32)
        
        for i, data_idx in enumerate(batch_indices):
            img = self.x_data[data_idx].squeeze()
            label = self.y_data[data_idx]
            
            if label == 4 and np.random.rand() < self.prob_dilate_4:
                img = dilate_digit(img)
            elif label == 9 and np.random.rand() < self.prob_erode_9:
                img = erode_digit(img)
            
            batch_x[i] = (img / 255.0).reshape(28, 28, 1)
            batch_y[i] = label
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def load_and_preprocess_data():
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Keep as uint8 for augmentation, reshape to add channel dimension
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # Normalize test data (train data normalized in generator)
    x_test = x_test.astype('float32') / 255.0
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    return (x_train, y_train), (x_test, y_test)

def build_baseline_cnn():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

class TrainingLogger(keras.callbacks.Callback):
    def __init__(self):
        self.training_log = []
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': logs.get('loss'),
            'train_accuracy': logs.get('accuracy'),
            'val_loss': logs.get('val_loss'),
            'val_accuracy': logs.get('val_accuracy')
        }
        self.training_log.append(epoch_data)
        print(f"Epoch {epoch + 1}: loss={logs.get('loss'):.4f}, acc={logs.get('accuracy'):.4f}, "
              f"val_loss={logs.get('val_loss'):.4f}, val_acc={logs.get('val_accuracy'):.4f}")

def train_augmented_model(model, train_data, test_data, seed, epochs=10, augmentation_prob=0.12):
    x_train, y_train = train_data
    x_test, y_test = test_data
    
    print(f"\n{'='*50}")
    print(f"Training augmented model with seed: {seed}")
    print(f"Augmentation probability for 4s: {augmentation_prob}")
    print(f"{'='*50}")
    
    # Create augmented data generator
    train_generator = AugmentedDataGenerator(
        x_train, y_train,
        batch_size=128,
        prob_dilate_4=0.12,
        prob_erode_9=0.12,
        shuffle=True
    )

    logger = TrainingLogger()
    
    start_time = datetime.now()
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[logger],
        verbose=0
    )
    end_time = datetime.now()
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate per-class accuracy
    per_class_accuracy = []
    for class_idx in range(10):
        class_mask = (y_test == class_idx)
        if np.sum(class_mask) > 0:
            class_acc = np.mean(y_pred_classes[class_mask] == y_test[class_mask])
            per_class_accuracy.append(class_acc)
        else:
            per_class_accuracy.append(0.0)
    
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    
    results = {
        'seed': seed,
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'per_class_accuracy': [float(acc) for acc in per_class_accuracy],
        'training_time_seconds': (end_time - start_time).total_seconds(),
        'confusion_matrix': conf_matrix.tolist(),
        'model_params': model.count_params(),
        'epochs': epochs,
        'augmentation_prob': augmentation_prob,
        'training_log': logger.training_log
    }
    
    return results, model, y_pred, y_pred_classes

def analyze_4_to_9_errors(y_true, y_pred_classes):
    """Calculate specific metrics for 4→9 confusion"""
    # Find all true 4s
    true_4s_mask = (y_true == 4)
    true_4s_indices = np.where(true_4s_mask)[0]
    
    # Count how many true 4s were predicted as 9
    four_to_nine_errors = np.sum((y_true == 4) & (y_pred_classes == 9))
    total_fours = np.sum(y_true == 4)
    
    # Calculate error rate
    four_to_nine_rate = four_to_nine_errors / total_fours if total_fours > 0 else 0
    
    # Calculate precision and recall for classes 4 and 9
    precision_4 = np.sum((y_pred_classes == 4) & (y_true == 4)) / np.sum(y_pred_classes == 4)
    recall_4 = np.sum((y_pred_classes == 4) & (y_true == 4)) / np.sum(y_true == 4)
    
    precision_9 = np.sum((y_pred_classes == 9) & (y_true == 9)) / np.sum(y_pred_classes == 9)
    recall_9 = np.sum((y_pred_classes == 9) & (y_true == 9)) / np.sum(y_true == 9)
    
    return {
        'four_to_nine_errors': int(four_to_nine_errors),
        'total_fours': int(total_fours),
        'four_to_nine_rate': float(four_to_nine_rate),
        'precision_4': float(precision_4),
        'recall_4': float(recall_4),
        'precision_9': float(precision_9),
        'recall_9': float(recall_9)
    }

def compare_with_baseline(baseline_results_file, augmented_results):
    """Compare augmented results with baseline"""
    # Load baseline results
    with open(baseline_results_file, 'r') as f:
        baseline_results = json.load(f)
    
    baseline_confusion = np.array(baseline_results['confusion_matrix'])
    augmented_confusion = np.array(augmented_results['confusion_matrix'])
    
    # Calculate 4→9 errors for both
    baseline_4_to_9 = baseline_confusion[4, 9]  # row=true, col=predicted
    augmented_4_to_9 = augmented_confusion[4, 9]
    
    # Calculate reduction
    reduction = (baseline_4_to_9 - augmented_4_to_9) / baseline_4_to_9 if baseline_4_to_9 > 0 else 0
    
    comparison = {
        'baseline_4_to_9_errors': int(baseline_4_to_9),
        'augmented_4_to_9_errors': int(augmented_4_to_9),
        'reduction_count': int(baseline_4_to_9 - augmented_4_to_9),
        'reduction_percentage': float(reduction * 100),
        'baseline_accuracy': baseline_results['test_accuracy'],
        'augmented_accuracy': augmented_results['test_accuracy'],
        'accuracy_change': augmented_results['test_accuracy'] - baseline_results['test_accuracy']
    }
    
    return comparison

def run_augmented_experiment(seed=42, epochs=10, augmentation_prob=0.12):
    """Run the targeted augmentation experiment"""
    
    output_dir = 'mnist_augmented_results'
    os.makedirs(output_dir, exist_ok=True)
    
    set_seeds(seed)
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Build and train augmented model
    model = build_baseline_cnn()
    print("\nModel Architecture:")
    model.summary()
    
    results, trained_model, y_pred, y_pred_classes = train_augmented_model(
        model, (x_train, y_train), (x_test, y_test), seed, epochs, augmentation_prob
    )
    
    # Analyze 4→9 specific errors
    error_analysis = analyze_4_to_9_errors(y_test, y_pred_classes)
    results.update(error_analysis)
    
    # Save results
    result_file = os.path.join(output_dir, f'augmented_results_seed_{seed}.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    model_file = os.path.join(output_dir, f'augmented_model_seed_{seed}.weights.h5')
    trained_model.save_weights(model_file)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(np.array(results['confusion_matrix']), annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title(f'Augmented Model Confusion Matrix (Seed: {seed})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, f'augmented_confusion_seed_{seed}.png'), 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    # Compare with baseline if available
    baseline_file = 'mnist_baseline_results/results_seed_42.json'
    if os.path.exists(baseline_file):
        comparison = compare_with_baseline(baseline_file, results)
        
        comparison_file = os.path.join(output_dir, f'comparison_seed_{seed}.json')
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n{'='*60}")
        print("BASELINE vs AUGMENTED COMPARISON")
        print(f"{'='*60}")
        print(f"Baseline 4→9 errors: {comparison['baseline_4_to_9_errors']}")
        print(f"Augmented 4→9 errors: {comparison['augmented_4_to_9_errors']}")
        print(f"Error reduction: {comparison['reduction_count']} ({comparison['reduction_percentage']:.1f}%)")
        print(f"Overall accuracy change: {comparison['accuracy_change']:.4f}")
        
        if comparison['reduction_percentage'] >= 30:
            print("\nSUCCESS: Achieved ≥30% reduction in 4→9 errors!")
        else:
            print(f"\nTarget not met: {comparison['reduction_percentage']:.1f}% < 30% target")
    
    print(f"\nAugmented Model Results:")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"4→9 Error Rate: {results['four_to_nine_rate']:.4f}")
    print(f"Precision (4): {results['precision_4']:.4f}")
    print(f"Recall (4): {results['recall_4']:.4f}")
    print(f"Precision (9): {results['precision_9']:.4f}")
    print(f"Recall (9): {results['recall_9']:.4f}")
    
    return results

if __name__ == "__main__":
    # Run the augmented experiment
    results = run_augmented_experiment(seed=42, epochs=10, augmentation_prob=0.12)