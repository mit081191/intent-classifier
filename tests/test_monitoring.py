from intent_classifier.monitoring.generate_monitoring_report import (
    determine_recommendation,
)


# ---------------------------------------------------------
# 1. Stable model -> NO_ACTION
# ---------------------------------------------------------

def test_no_action_when_all_signals_are_low():
    """
    If input drift, prediction drift and performance drift
    are all LOW, no intervention should be required.
    """

    result = determine_recommendation(
        input_drift_status="LOW",
        prediction_drift_status="LOW",
        performance_status="LOW",
    )

    assert result == "NO_ACTION"


# ---------------------------------------------------------
# 2. Moderate input drift -> REVIEW
# ---------------------------------------------------------

def test_review_when_input_drift_is_moderate():
    """
    Moderate input-language drift should trigger review,
    even when model performance has not yet degraded.
    """

    result = determine_recommendation(
        input_drift_status="MODERATE",
        prediction_drift_status="LOW",
        performance_status="LOW",
    )

    assert result == "REVIEW"


# ---------------------------------------------------------
# 3. Moderate prediction drift -> REVIEW
# ---------------------------------------------------------

def test_review_when_prediction_drift_is_moderate():
    """
    Changed prediction distributions should trigger
    investigation before automatic retraining.
    """

    result = determine_recommendation(
        input_drift_status="LOW",
        prediction_drift_status="MODERATE",
        performance_status="LOW",
    )

    assert result == "REVIEW"


# ---------------------------------------------------------
# 4. Performance degradation alone -> REVIEW
# ---------------------------------------------------------

def test_review_when_only_performance_is_high():
    """
    A high performance drop without supporting input or
    prediction drift should first trigger investigation.

    This avoids blindly retraining from one signal.
    """

    result = determine_recommendation(
        input_drift_status="LOW",
        prediction_drift_status="LOW",
        performance_status="HIGH",
    )

    assert result == "REVIEW"


# ---------------------------------------------------------
# 5. High input drift + high performance drop -> RETRAIN
# ---------------------------------------------------------

def test_retrain_when_input_and_performance_drift_are_high():
    """
    Strong language drift together with confirmed model
    degradation is sufficient evidence to recommend retraining.
    """

    result = determine_recommendation(
        input_drift_status="HIGH",
        prediction_drift_status="MODERATE",
        performance_status="HIGH",
    )

    assert result == "RETRAIN"


# ---------------------------------------------------------
# 6. High prediction drift + high performance drop -> RETRAIN
# ---------------------------------------------------------

def test_retrain_when_prediction_and_performance_drift_are_high():
    """
    Confirmed performance degradation plus major changes
    in model prediction behaviour should recommend retraining.
    """

    result = determine_recommendation(
        input_drift_status="MODERATE",
        prediction_drift_status="HIGH",
        performance_status="HIGH",
    )

    assert result == "RETRAIN"