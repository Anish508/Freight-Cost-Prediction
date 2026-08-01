import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ---------------------------------------------------------------------------
# Model training helpers
# ---------------------------------------------------------------------------

def train_logistic_regression(
    X_train,
    y_train,
    random_state: int = 42,
    **kwargs,
) -> LogisticRegression:
    """
    Fit and return a Logistic Regression classifier.

    Parameters
    ----------
    X_train      : Scaled training feature matrix.
    y_train      : Training labels.
    random_state : Reproducibility seed.
    **kwargs     : Any additional keyword arguments forwarded to
                   ``LogisticRegression``.
    """
    model = LogisticRegression(random_state=random_state, **kwargs)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(
    X_train,
    y_train,
    random_state: int = 42,
    **kwargs,
) -> DecisionTreeClassifier:
    """
    Fit and return a Decision Tree classifier.

    Parameters
    ----------
    X_train      : Scaled training feature matrix.
    y_train      : Training labels.
    random_state : Reproducibility seed.
    **kwargs     : Any additional keyword arguments forwarded to
                   ``DecisionTreeClassifier``.
    """
    model = DecisionTreeClassifier(random_state=random_state, **kwargs)
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train,
    y_train,
    random_state: int = 42,
    **kwargs,
) -> RandomForestClassifier:
    """
    Fit and return a Random Forest classifier.

    Parameters
    ----------
    X_train      : Scaled training feature matrix.
    y_train      : Training labels.
    random_state : Reproducibility seed.
    **kwargs     : Any additional keyword arguments forwarded to
                   ``RandomForestClassifier``.
    """
    model = RandomForestClassifier(random_state=random_state, **kwargs)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_random_forest(
    X_train,
    y_train,
    param_grid: dict | None = None,
    cv: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> GridSearchCV:
    """
    Run a GridSearchCV over ``RandomForestClassifier`` and return the fitted
    ``GridSearchCV`` object.

    The best estimator is accessible via ``grid_search.best_estimator_``.

    Parameters
    ----------
    X_train      : Scaled training feature matrix.
    y_train      : Training labels.
    param_grid   : Hyperparameter grid.  Uses the notebook defaults when
                   ``None``.
    cv           : Number of cross-validation folds.
    random_state : Seed for the ``RandomForestClassifier``.
    n_jobs       : Parallelism level for both the model and grid search.
    """
    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 4, 5, 6],
            "min_samples_split": [2, 3, 5],
            "min_samples_leaf": [1, 2, 5],
            "criterion": ["gini", "entropy"],
        }

    from sklearn.metrics import make_scorer

    rf = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
    scorer = make_scorer(f1_score)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        verbose=2,
        n_jobs=n_jobs,
    )
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best F1 Score  :", round(grid_search.best_score_, 4))

    return grid_search


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name: str = "") -> dict:
    """
    Print a classification report and confusion matrix for *model* on the
    given test set.  Also returns a summary dictionary.

    Parameters
    ----------
    model       : Fitted sklearn-compatible classifier.
    X_test      : Scaled test feature matrix.
    y_test      : True test labels.
    model_name  : Label used in the printed output.

    Returns
    -------
    dict with keys: model_name, accuracy, precision, recall, f1.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n{'=' * 50}")
    print(f"Model : {model_name}")
    print(f"{'=' * 50}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


def get_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Return a DataFrame of feature importances sorted in descending order.

    Parameters
    ----------
    model         : Fitted ``RandomForestClassifier``.
    feature_names : Names corresponding to the columns used during training.
    """
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    return importance_df


# ---------------------------------------------------------------------------
# Scaling helpers
# ---------------------------------------------------------------------------

def scale_features(
    X_train,
    X_test,
    scaler_type: str = "minmax",
):
    """
    Fit a scaler on *X_train* and transform both splits.

    Parameters
    ----------
    X_train      : Training feature matrix.
    X_test       : Test feature matrix.
    scaler_type  : ``"standard"`` for ``StandardScaler``, ``"minmax"`` for
                   ``MinMaxScaler`` (default).

    Returns
    -------
    X_train_scaled, X_test_scaled, fitted_scaler
    """
    if scaler_type == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
