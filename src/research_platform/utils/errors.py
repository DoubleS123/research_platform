"""Application-specific errors."""


class ResearchPlatformError(Exception):
    """Base error for the research platform."""


class DataLoadError(ResearchPlatformError):
    """Raised when CSV or dataset loading fails."""


class ValidationError(ResearchPlatformError):
    """Raised when input validation fails."""


class EstimationError(ResearchPlatformError):
    """Raised when a statistical model fails to fit."""


# Backward-compatible aliases
DataValidationError = ValidationError
