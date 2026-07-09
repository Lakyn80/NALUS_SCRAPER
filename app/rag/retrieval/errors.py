class RetrievalConfigurationError(RuntimeError):
    """Raised when the production retrieval stack is not configured safely."""


class RetrievalDependencyError(RuntimeError):
    """Raised when a required retrieval dependency is unavailable at query time."""
