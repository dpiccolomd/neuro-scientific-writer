"""Custom exceptions for quality control operations."""


class QualityControlError(Exception):
    """Base exception for quality control errors."""
    
    def __init__(self, message: str, validation_context: str = None):
        super().__init__(message)
        self.validation_context = validation_context
        
    def __str__(self):
        if self.validation_context:
            return f"Quality control error in {self.validation_context}: {super().__str__()}"
        return f"Quality control error: {super().__str__()}"


class ValidationError(QualityControlError):
    """Exception for validation failures."""
    pass


class CitationVerificationError(QualityControlError):
    """Exception for citation verification failures."""
    pass


class FactCheckingError(QualityControlError):
    """Exception for fact-checking failures."""
    pass


class StatisticalValidationError(QualityControlError):
    """Exception for statistical validation failures."""
    pass


class PlagiarismDetectionError(QualityControlError):
    """Exception for plagiarism detection failures."""
    pass


class CriticalValidationError(QualityControlError):
    """Exception for critical validation failures that prevent publication."""
    
    def __init__(self, message: str, severity: str = "CRITICAL"):
        super().__init__(message)
        self.severity = severity