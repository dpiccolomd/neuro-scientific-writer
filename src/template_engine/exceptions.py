"""Custom exceptions for template generation."""


class TemplateGenerationError(Exception):
    """Base exception for template generation errors."""
    
    def __init__(self, message: str, template_id: str = None):
        super().__init__(message)
        self.template_id = template_id
        
    def __str__(self):
        if self.template_id:
            return f"Template generation error for {self.template_id}: {super().__str__()}"
        return f"Template generation error: {super().__str__()}"


class TemplateValidationError(TemplateGenerationError):
    """Exception for template validation failures."""
    pass


class TemplateStructureError(TemplateGenerationError):
    """Exception for template structure definition errors."""
    pass


class TemplateRenderingError(TemplateGenerationError):
    """Exception for template rendering failures."""
    pass


class TemplatePatternError(TemplateGenerationError):
    """Exception for pattern-based template generation errors."""
    pass