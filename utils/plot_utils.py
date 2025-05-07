# Plotting and visualization
import matplotlib.pyplot as plt
import optuna.visualization as vis

# Scikit-learn metrics and visualizations
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_auc_score,
    roc_curve
)


def optuna_visualizations(study, params):
    # Visualize the hyperparameter combinations and their performance
    fig = vis.plot_contour(study, params=params)
    fig.show()

    # Visualize the optimization history
    fig = vis.plot_optimization_history(study)
    fig.show()

    # Visualize the parameter importance
    fig = vis.plot_param_importances(study)
    fig.show()


def precision_recall_plot(y_test, y_pred, title):
    # Calculate precision and recall
    precision, recall, _ = precision_recall_curve(y_test, y_pred)

    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve', color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()


def confusion_matrix_plot(y_test, y_pred, title):
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Fall", "No Fall"])
    disp.plot(cmap='Blues')
    disp.ax_.set_title(title)
    disp.ax_.set_xlabel("Predicted")
    disp.ax_.set_ylabel("True")
    disp.figure_.set_size_inches(8, 6)
    disp.figure_.tight_layout()
    plt.show()


def plot_training_history(history):
    plt.figure(figsize=(14, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def roc_curve_plot(y_test, y_proba, title):
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba, pos_label=1)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(roc_auc), color='blue')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
