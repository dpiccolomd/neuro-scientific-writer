"""Exceptions for citation management system."""


class CitationManagerError(Exception):
    """Base exception for citation management errors."""
    pass


class CitationExtractionError(CitationManagerError):
    """Raised when citation extraction fails."""
    pass


class ReferenceValidationError(CitationManagerError):
    """Raised when reference validation fails."""
    pass


class APAFormattingError(CitationManagerError):
    """Raised when APA formatting fails."""
    pass


class ZoteroIntegrationError(CitationManagerError):
    """Raised when Zotero integration fails."""
    pass


class InvalidCitationError(CitationManagerError):
    """Raised when citation format is invalid."""
    pass


class DuplicateReferenceError(CitationManagerError):
    """Raised when duplicate references are detected."""
    pass


class MissingReferenceError(CitationManagerError):
    """Raised when a referenced citation cannot be found."""
    pass