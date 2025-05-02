import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, \
    roc_curve, make_scorer
import joblib

# Load the preprocessed dataset from CSV files
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')


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


def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"=== {model_name} ===")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["ADL", "Fall"]))

    print("F1 Score: ")
    print(f1_score(y_test, y_pred, average='macro'))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

    return y_pred, y_proba


def roc_curve_plot(y_test, y_proba, title):
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1], pos_label=1)
    roc_auc = roc_auc_score(y_test, y_proba[:, 1])

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


def random_search(model, params, model_name):
    # Define scoring method
    f1_macro = make_scorer(f1_score, average='macro')

    # Randomized search for hyperparameter tuning
    model_random = RandomizedSearchCV(
        model,
        param_distributions=params,
        n_iter=8,
        scoring=f1_macro,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    # Fit the model
    model_random.fit(X_train, y_train.values.ravel())

    # Evaluate the best model
    y_pred, y_proba = evaluate_model(model_random.best_estimator_, X_test, y_test, model_name=model_name)

    return y_pred, y_proba, model_random.best_estimator_


def grid_search(model, params, model_name):
    # Define scoring method
    f1_macro = make_scorer(f1_score, average='macro')

    # Grid search for hyperparameter tuning
    model_grid = GridSearchCV(
        model,
        param_grid=params,
        scoring=f1_macro,
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    # Fit the model
    model_grid.fit(X_train, y_train.values.ravel())

    # Evaluate the best model
    y_pred, y_proba = evaluate_model(model_grid.best_estimator_, X_test, y_test, model_name=model_name)

    return y_pred, y_proba, model_grid.best_estimator_


def run_model(model, params, model_name, mode="grid"):
    print(f"Running {model_name} with {mode} search...")

    # Run hyperparameter tuning
    if mode == "random":
        y_pred, y_proba, best_estimator = random_search(model, params, model_name)
    else:
        y_pred, y_proba, best_estimator = grid_search(model, params, model_name)

    # Plot confusion matrix
    confusion_matrix_plot(y_test, y_pred, f"Confusion Matrix for {model_name}")

    # Plot ROC curve
    roc_curve_plot(y_test, y_proba, f"ROC Curve for {model_name}")

    # Save the model
    joblib.dump(best_estimator.best_estimator_, f'{model_name.lower()}_model.pkl')


def run_svm():
    # Define SVM parameters for RandomizedSearchCV
    svm_params = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    # Create SVM model
    svm_model = SVC(probability=True)

    # Run SVM model with hyperparameter tuning
    run_model(svm_model, svm_params, "SVM")


def run_rf():
    # Define Random Forest parameters for RandomizedSearchCV
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Create Random Forest model
    rf_model = RandomForestClassifier()

    # Run Random Forest model with hyperparameter tuning
    run_model(rf_model, rf_params, "Random Forest")


def run_knn():
    # Define KNN parameters for RandomizedSearchCV
    knn_params = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    # Create KNN model
    knn_model = KNeighborsClassifier()

    # Run KNN model with hyperparameter tuning
    run_model(knn_model, knn_params, "KNN")


if __name__ == "__main__":
    # Run all models
    run_svm()
    run_rf()
    run_knn()
