"""Custom exceptions for PDF processing."""


class PDFProcessingError(Exception):
    """Base exception for PDF processing errors."""
    
    def __init__(self, message: str, file_path: str = None):
        super().__init__(message)
        self.file_path = file_path
        
    def __str__(self):
        if self.file_path:
            return f"PDF processing error for {self.file_path}: {super().__str__()}"
        return super().__str__()


class UnsupportedPDFError(PDFProcessingError):
    """Exception for unsupported PDF formats or corrupted files."""
    pass


class ExtractionError(PDFProcessingError):
    """Exception for text extraction failures."""
    pass


class StructureDetectionError(PDFProcessingError):
    """Exception for document structure detection failures."""
    pass


class CitationExtractionError(PDFProcessingError):
    """Exception for citation extraction failures."""
    pass