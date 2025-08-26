"""Data models for template generation."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime


class TemplateType(Enum):
    """Types of templates that can be generated."""
    INTRODUCTION_FUNNEL = "introduction_funnel"
    INTRODUCTION_DIRECT = "introduction_direct"
    LITERATURE_REVIEW = "literature_review"
    HYPOTHESIS_DRIVEN = "hypothesis_driven"
    METHODOLOGY_FOCUSED = "methodology_focused"
    CUSTOM = "custom"


class ParagraphType(Enum):
    """Types of paragraphs in neuroscience introductions."""
    BROAD_CONTEXT = "broad_context"
    BACKGROUND_LITERATURE = "background_literature"
    GAP_IDENTIFICATION = "gap_identification"
    HYPOTHESIS_STATEMENT = "hypothesis_statement"
    STUDY_OBJECTIVES = "study_objectives"
    SIGNIFICANCE_STATEMENT = "significance_statement"
    METHODOLOGY_PREVIEW = "methodology_preview"
    TRANSITION = "transition"


@dataclass
class TemplateVariable:
    """Represents a variable placeholder in the template."""
    name: str
    description: str
    variable_type: str  # text, citation, list, number
    default_value: Optional[str] = None
    required: bool = True
    example: Optional[str] = None
    validation_pattern: Optional[str] = None
    
    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("Variable name cannot be empty")
        if self.variable_type not in ['text', 'citation', 'list', 'number']:
            raise ValueError("Invalid variable type")


@dataclass
class ParagraphTemplate:
    """Template for a single paragraph within an introduction."""
    paragraph_type: ParagraphType
    title: str
    content_template: str
    variables: List[TemplateVariable]
    guidance_notes: List[str]
    example_sentences: List[str]
    min_length: int = 50  # minimum words
    max_length: int = 200  # maximum words
    citation_density: float = 0.2  # expected citations per sentence
    
    def __post_init__(self):
        if not self.content_template.strip():
            raise ValueError("Content template cannot be empty")
        if self.min_length < 0 or self.max_length < self.min_length:
            raise ValueError("Invalid length constraints")
        if not 0.0 <= self.citation_density <= 1.0:
            raise ValueError("Citation density must be between 0.0 and 1.0")


@dataclass
class SectionTemplate:
    """Template for a complete section (e.g., full introduction)."""
    section_type: str
    title: str
    description: str
    paragraphs: List[ParagraphTemplate]
    structure_notes: List[str]
    flow_guidelines: List[str]
    estimated_word_count: int
    
    def __post_init__(self):
        if not self.paragraphs:
            raise ValueError("Section must have at least one paragraph template")
        if self.estimated_word_count < 0:
            raise ValueError("Estimated word count cannot be negative")


@dataclass
class TemplateMetadata:
    """Metadata about the generated template."""
    template_id: str
    template_type: TemplateType
    source_patterns: List[str]  # patterns used to generate template
    confidence_score: float
    complexity_level: str  # basic, intermediate, advanced
    target_audience: str
    field_specialization: str
    generated_from_papers: int  # number of source papers
    pattern_coverage: float  # how well patterns are covered
    
    def __post_init__(self):
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        if not 0.0 <= self.pattern_coverage <= 1.0:
            raise ValueError("Pattern coverage must be between 0.0 and 1.0")


@dataclass
class TemplateSection:
    """A rendered section of the template with content."""
    section_template: SectionTemplate
    rendered_content: str
    filled_variables: Dict[str, Any]
    unfilled_variables: List[str]
    word_count: int
    quality_score: float
    
    def __post_init__(self):
        if not self.rendered_content.strip():
            raise ValueError("Rendered content cannot be empty")
        if self.word_count < 0:
            self.word_count = len(self.rendered_content.split())


@dataclass
class GeneratedTemplate:
    """Complete generated template with all sections and metadata."""
    template_id: str
    metadata: TemplateMetadata
    sections: List[TemplateSection]
    global_variables: List[TemplateVariable]
    style_guidelines: Dict[str, Any]
    quality_metrics: Dict[str, float]
    generated_at: datetime
    generator_version: str = "1.0.0"
    
    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = datetime.now()
        if not self.sections:
            raise ValueError("Template must have at least one section")
    
    @property
    def total_word_count(self) -> int:
        """Get total estimated word count for the template."""
        return sum(section.word_count for section in self.sections)
    
    @property
    def completion_percentage(self) -> float:
        """Get percentage of variables that have been filled."""
        total_vars = len(self.global_variables)
        for section in self.sections:
            total_vars += len(section.section_template.paragraphs) * 2  # Approximate
        
        filled_vars = sum(len(section.filled_variables) for section in self.sections)
        return (filled_vars / total_vars * 100) if total_vars > 0 else 0.0
    
    @property
    def overall_quality_score(self) -> float:
        """Get overall quality score for the template."""
        if not self.sections:
            return 0.0
        return sum(section.quality_score for section in self.sections) / len(self.sections)
    
    def get_unfilled_variables(self) -> List[str]:
        """Get all unfilled variables across sections."""
        unfilled = []
        for section in self.sections:
            unfilled.extend(section.unfilled_variables)
        return list(set(unfilled))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "template_id": self.template_id,
            "metadata": {
                **self.metadata.__dict__,
                "template_type": self.metadata.template_type.value
            },
            "sections": [
                {
                    "section_template": {
                        **section.section_template.__dict__,
                        "paragraphs": [
                            {
                                **para.__dict__,
                                "paragraph_type": para.paragraph_type.value,
                                "variables": [var.__dict__ for var in para.variables]
                            }
                            for para in section.section_template.paragraphs
                        ]
                    },
                    "rendered_content": section.rendered_content,
                    "filled_variables": section.filled_variables,
                    "unfilled_variables": section.unfilled_variables,
                    "word_count": section.word_count,
                    "quality_score": section.quality_score
                }
                for section in self.sections
            ],
            "global_variables": [var.__dict__ for var in self.global_variables],
            "style_guidelines": self.style_guidelines,
            "quality_metrics": self.quality_metrics,
            "generated_at": self.generated_at.isoformat(),
            "generator_version": self.generator_version,
            # Computed properties
            "total_word_count": self.total_word_count,
            "completion_percentage": self.completion_percentage,
            "overall_quality_score": self.overall_quality_score
        }