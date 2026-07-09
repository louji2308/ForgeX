class ForgeXError(Exception):
    """Base class for all ForgeX-specific exceptions."""

class DataValidationError(ForgeXError):
    """Synthetic or real data failed a structural/integrity check."""

class FeatureBuildError(ForgeXError):
    """A feature could not be computed for a scoring request."""

class ModelSchemaError(ForgeXError):
    """Loaded model's feature contract doesn't match what the caller expects."""

class FairnessGateFailure(ForgeXError):
    """A model or decision failed the mandatory fairness check."""

class WebhookAuthError(ForgeXError):
    """Inbound PM-platform webhook failed signature or auth verification."""

class MissingDependencyError(ForgeXError):
    """An optional dependency (spaCy, torch, LLM client) is not installed."""
