from .template_generator import TemplateGenerator
from .targeted_template_generator import TargetedTemplateGenerator
from .research_specification import ComprehensiveResearchSpecification as ResearchSpecification
from .models import (
    GeneratedTemplate, TemplateSection, SectionTemplate, ParagraphTemplate,
    TemplateMetadata, TemplateType, ParagraphType, TemplateVariable
)
from .exceptions import TemplateGenerationError, TemplateValidationError

__all__ = [
    "TemplateGenerator",
    "TargetedTemplateGenerator",
    "ResearchSpecification",
    "SectionTemplate",
    "ParagraphTemplate",
    "GeneratedTemplate",
    "TemplateSection",
    "TemplateMetadata",
    "TemplateType",
    "ParagraphType", 
    "TemplateVariable",
    "TemplateGenerationError",
    "TemplateValidationError"
]