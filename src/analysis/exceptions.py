"""Custom exceptions for NLP analysis components."""


class AnalysisError(Exception):
    """Base exception for text analysis errors."""
    
    def __init__(self, message: str, text_snippet: str = None):
        super().__init__(message)
        self.text_snippet = text_snippet
        
    def __str__(self):
        if self.text_snippet:
            return f"Analysis error: {super().__str__()}\nText: {self.text_snippet[:100]}..."
        return f"Analysis error: {super().__str__()}"


class ModelLoadError(AnalysisError):
    """Exception for NLP model loading failures."""
    pass


class PatternDetectionError(AnalysisError):
    """Exception for writing pattern detection failures."""
    pass


class TerminologyExtractionError(AnalysisError):
    """Exception for neuroscience terminology extraction failures."""
    pass


class SentenceAnalysisError(AnalysisError):
    """Exception for sentence structure analysis failures."""
    pass