import pytest
# add necessary import
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics
from ml.data import apply_label

# Test 1: train_model returns RandomForestClassifier
def test_train_model():
    """
    # this test is to ensure the proper model is returned
    """
    X_train = np.array([[1,2], [3,4], [5,6], [7,8], [9,10], [11,12]])
    y_train = np.array([0, 1, 0, 1, 0, 1])
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

# Test 2: compute_model_metrics returns expected values
def test_compute_model_metrics():
    """
    # this test is to ensure that the predicition model
    will return expected values for a perfect prediction
    """
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 1, 0, 0, 1])
    precision_score, recall_score, fbeta_score = compute_model_metrics(y_true, y_pred)
    assert precision_score == pytest.approx(1.0)
    assert recall_score == pytest.approx(1.0)
    assert fbeta_score == pytest.approx(1.0)

# Test 3: apply_label returns correct string
def test_apply_label():
    """
    # this test is to ensure the apply_label returns
    the expected strings
    """
    assert apply_label([0]) == "<=50K"
    assert apply_label([1]) == ">50K"
