import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Set random seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Load and preprocess MNIST data
def load_and_preprocess_data():
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape to add channel dimension (28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    return (x_train, y_train), (x_test, y_test)

# Build the baseline CNN architecture
def build_baseline_cnn():
    model = keras.Sequential([
        # Conv Layer 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Conv Layer 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Custom callback to save training logs
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

# Train the model
def train_baseline_model(model, train_data, test_data, seed, epochs=10):
    x_train, y_train = train_data
    x_test, y_test = test_data
    
    print(f"\n{'='*50}")
    print(f"Training model with seed: {seed}")
    print(f"{'='*50}")
    
    # Create training logger
    logger = TrainingLogger()
    
    # Train the model
    start_time = datetime.now()
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[logger],
        verbose=0
    )
    end_time = datetime.now()
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    # Get predictions for confusion matrix
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
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    
    # Compile results
    results = {
        'seed': seed,
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'per_class_accuracy': [float(acc) for acc in per_class_accuracy],
        'training_time_seconds': (end_time - start_time).total_seconds(),
        'confusion_matrix': conf_matrix.tolist(),
        'model_params': model.count_params(),
        'epochs': epochs,
        'training_log': logger.training_log
    }
    
    return results, model, y_pred, y_pred_classes

# Plot training curves
def plot_training_curves(training_log, seed, save_path=None):
    df = pd.DataFrame(training_log)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    ax1.plot(df['epoch'], df['train_loss'], 'b-', label='Training Loss')
    ax1.plot(df['epoch'], df['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title(f'Model Loss (Seed: {seed})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(df['epoch'], df['train_accuracy'], 'b-', label='Training Accuracy')
    ax2.plot(df['epoch'], df['val_accuracy'], 'r-', label='Validation Accuracy')
    ax2.set_title(f'Model Accuracy (Seed: {seed})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(conf_matrix, seed, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title(f'Confusion Matrix (Seed: {seed})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# Save misclassified examples
def save_misclassified_examples(x_test, y_test, y_pred_classes, seed, num_examples=20):
    misclassified_indices = np.where(y_test != y_pred_classes)[0]
    
    if len(misclassified_indices) == 0:
        print("No misclassified examples found!")
        return []
    
    # Sample some misclassified examples
    sample_size = min(num_examples, len(misclassified_indices))
    sample_indices = np.random.choice(misclassified_indices, sample_size, replace=False)
    
    misclassified_data = []
    for idx in sample_indices:
        misclassified_data.append({
            'index': int(idx),
            'true_label': int(y_test[idx]),
            'predicted_label': int(y_pred_classes[idx]),
            'image': x_test[idx].squeeze()
        })
    
    # Plot some examples
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, data in enumerate(misclassified_data[:20]):
        axes[i].imshow(data['image'], cmap='gray')
        axes[i].set_title(f'True: {data["true_label"]}, Pred: {data["predicted_label"]}')
        axes[i].axis('off')
    
    plt.suptitle(f'Misclassified Examples (Seed: {seed})')
    plt.tight_layout()
    plt.show()
    
    return misclassified_data

# Main execution function
def run_baseline_experiment(seeds=[42, 123, 456], epochs=10):
    """Run the complete baseline experiment with multiple seeds"""
    
    # Create output directory
    output_dir = 'mnist_baseline_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data once
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    all_results = []
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {i+1}/{len(seeds)} - SEED: {seed}")
        print(f"{'='*60}")
        
        # Set seeds
        set_seeds(seed)
        
        # Build model
        model = build_baseline_cnn()
        
        # Print model summary for first run only
        if i == 0:
            print("\nModel Architecture:")
            model.summary()
        
        # Train model
        results, trained_model, y_pred, y_pred_classes = train_baseline_model(
            model, (x_train, y_train), (x_test, y_test), seed, epochs
        )
        
        all_results.append(results)
        
        # Save individual results
        result_file = os.path.join(output_dir, f'results_seed_{seed}.json')
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save model weights
        model_file = os.path.join(output_dir, f'model_seed_{seed}.weights.h5')
        trained_model.save_weights(model_file)
        
        # Plot training curves
        plot_path = os.path.join(output_dir, f'training_curves_seed_{seed}.png')
        plot_training_curves(results['training_log'], seed, plot_path)
        
        # Plot confusion matrix
        conf_path = os.path.join(output_dir, f'confusion_matrix_seed_{seed}.png')
        plot_confusion_matrix(np.array(results['confusion_matrix']), seed, conf_path)
        
        # Save misclassified examples for first run
        if i == 0:
            misclassified = save_misclassified_examples(x_test, y_test, y_pred_classes, seed)
        
        print(f"\nResults for seed {seed}:")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Test Loss: {results['test_loss']:.4f}")
        print(f"Training Time: {results['training_time_seconds']:.1f} seconds")
        print(f"Total Parameters: {results['model_params']:,}")
    
    # Aggregate results across seeds
    accuracies = [r['test_accuracy'] for r in all_results]
    losses = [r['test_loss'] for r in all_results]
    
    summary = {
        'mean_accuracy': float(np.mean(accuracies)),
        'std_accuracy': float(np.std(accuracies)),
        'mean_loss': float(np.mean(losses)),
        'std_loss': float(np.std(losses)),
        'individual_results': all_results,
        'experiment_timestamp': datetime.now().isoformat()
    }
    
    # Save summary
    summary_file = os.path.join(output_dir, 'experiment_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Mean Test Accuracy: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
    print(f"Mean Test Loss: {summary['mean_loss']:.4f} ± {summary['std_loss']:.4f}")
    print(f"All accuracies: {[f'{acc:.4f}' for acc in accuracies]}")
    print(f"Results saved in: {output_dir}/")
    
    return summary

if __name__ == "__main__":
    # Run the baseline experiment
    summary = run_baseline_experiment(seeds=[42, 123, 456], epochs=10)
    
    # Check if we hit our target
    target_accuracy = 0.975
    if summary['mean_accuracy'] >= target_accuracy:
        print(f"\nSUCCESS: Mean accuracy {summary['mean_accuracy']:.4f} meets target of {target_accuracy:.1%}")
    else:
        print(f"\nBELOW TARGET: Mean accuracy {summary['mean_accuracy']:.4f} below target of {target_accuracy:.1%}")