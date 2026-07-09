from forgex.errors import (
    ForgeXError,
    DataValidationError,
    FeatureBuildError,
    ModelSchemaError,
    FairnessGateFailure,
    WebhookAuthError,
    MissingDependencyError,
)


def test_exception_hierarchy():
    assert issubclass(DataValidationError, ForgeXError)
    assert issubclass(FeatureBuildError, ForgeXError)
    assert issubclass(ModelSchemaError, ForgeXError)
    assert issubclass(FairnessGateFailure, ForgeXError)
    assert issubclass(WebhookAuthError, ForgeXError)
    assert issubclass(MissingDependencyError, ForgeXError)


def test_data_validation_error():
    err = DataValidationError("test message")
    assert str(err) == "test message"
    assert isinstance(err, ForgeXError)


def test_fairness_gate_failure():
    err = FairnessGateFailure("Model failed fairness check")
    assert "fairness" in str(err).lower()
