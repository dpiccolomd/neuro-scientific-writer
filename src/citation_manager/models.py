"""Data models for citation management."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class CitationType(Enum):
    """Types of citations."""
    JOURNAL_ARTICLE = "journal_article"
    BOOK = "book"
    BOOK_CHAPTER = "book_chapter"
    CONFERENCE_PAPER = "conference_paper"
    THESIS = "thesis"
    PREPRINT = "preprint"
    WEB_RESOURCE = "web_resource"
    DATASET = "dataset"


class CitationStyle(Enum):
    """Supported citation styles."""
    APA = "apa"
    AMA = "ama"
    VANCOUVER = "vancouver"


class ValidationStatus(Enum):
    """Citation validation status."""
    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"
    UNVERIFIED = "unverified"


@dataclass
class Author:
    """Author information."""
    last_name: str
    first_name: Optional[str] = None
    middle_initial: Optional[str] = None
    suffix: Optional[str] = None
    orcid: Optional[str] = None
    
    def __post_init__(self):
        if not self.last_name.strip():
            raise ValueError("Author last name cannot be empty")
    
    @property
    def full_name(self) -> str:
        """Get formatted full name."""
        parts = [self.last_name]
        if self.first_name:
            parts.append(f"{self.first_name[0]}.")
        if self.middle_initial:
            parts.append(f"{self.middle_initial}.")
        return " ".join(parts)
    
    @property
    def apa_format(self) -> str:
        """Get APA formatted name."""
        name_parts = [self.last_name + ","]
        if self.first_name:
            name_parts.append(f"{self.first_name[0]}.")
        if self.middle_initial:
            name_parts.append(f"{self.middle_initial}.")
        return " ".join(name_parts)


@dataclass
class Journal:
    """Journal information."""
    name: str
    abbreviation: Optional[str] = None
    issn: Optional[str] = None
    impact_factor: Optional[float] = None
    publisher: Optional[str] = None
    
    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("Journal name cannot be empty")


@dataclass
class Reference:
    """Complete reference information."""
    reference_id: str
    citation_type: CitationType
    authors: List[Author]
    title: str
    year: int
    
    # Journal-specific fields
    journal: Optional[Journal] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    
    # Book-specific fields  
    publisher: Optional[str] = None
    place: Optional[str] = None
    edition: Optional[str] = None
    
    # Additional metadata
    abstract: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    url: Optional[str] = None
    accessed_date: Optional[datetime] = None
    language: str = "en"
    
    def __post_init__(self):
        if not self.title.strip():
            raise ValueError("Reference title cannot be empty")
        if not self.authors:
            raise ValueError("Reference must have at least one author")
        if self.year < 1800 or self.year > datetime.now().year + 1:
            raise ValueError(f"Invalid year: {self.year}")
    
    @property
    def first_author_surname(self) -> str:
        """Get first author's surname."""
        return self.authors[0].last_name if self.authors else ""
    
    @property
    def is_journal_article(self) -> bool:
        """Check if this is a journal article."""
        return self.citation_type == CitationType.JOURNAL_ARTICLE
    
    @property
    def short_citation(self) -> str:
        """Get short in-text citation format."""
        author_part = self.first_author_surname
        if len(self.authors) > 1:
            author_part += " et al." if len(self.authors) > 2 else f" & {self.authors[1].last_name}"
        return f"({author_part}, {self.year})"


@dataclass
class Citation:
    """In-text citation instance."""
    citation_id: str
    reference_id: str
    page_number: Optional[str] = None
    prefix: Optional[str] = None  # e.g., "see", "cf."
    suffix: Optional[str] = None  # e.g., "for review"
    is_parenthetical: bool = True
    position_in_text: int = 0
    surrounding_context: str = ""
    
    def __post_init__(self):
        if not self.reference_id.strip():
            raise ValueError("Citation must reference a valid reference_id")


@dataclass
class CitationContext:
    """Context information for a citation."""
    citation: Citation
    sentence: str
    paragraph: str
    section: str
    semantic_role: str  # supporting, contrasting, methodological, etc.
    confidence_score: float = 0.0
    
    def __post_init__(self):
        if not 0 <= self.confidence_score <= 1:
            raise ValueError("Confidence score must be between 0 and 1")


@dataclass
class ValidationResult:
    """Result of citation validation."""
    citation_id: str
    status: ValidationStatus
    confidence: float
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
    
    @property
    def is_valid(self) -> bool:
        """Check if citation is valid."""
        return self.status == ValidationStatus.VALID
    
    @property  
    def has_warnings(self) -> bool:
        """Check if citation has warnings."""
        return bool(self.warnings) or self.status == ValidationStatus.WARNING
    
    @property
    def has_errors(self) -> bool:
        """Check if citation has errors."""
        return bool(self.issues) or self.status == ValidationStatus.ERROR


@dataclass
class BibliographyEntry:
    """Formatted bibliography entry."""
    reference_id: str
    formatted_text: str
    style: CitationStyle
    sort_key: str
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.formatted_text.strip():
            raise ValueError("Bibliography entry cannot be empty")


@dataclass
class CitationReport:
    """Comprehensive citation analysis report."""
    total_citations: int
    unique_references: int
    validation_results: List[ValidationResult]
    style_consistency_score: float
    completeness_score: float
    accuracy_score: float
    recommendations: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    
    @property
    def valid_citations(self) -> int:
        """Count valid citations."""
        return sum(1 for vr in self.validation_results if vr.is_valid)
    
    @property
    def citations_with_warnings(self) -> int:
        """Count citations with warnings."""
        return sum(1 for vr in self.validation_results if vr.has_warnings)
    
    @property
    def citations_with_errors(self) -> int:
        """Count citations with errors."""
        return sum(1 for vr in self.validation_results if vr.has_errors)
    
    @property
    def overall_quality_score(self) -> float:
        """Calculate overall citation quality score."""
        if not self.validation_results:
            return 0.0
        
        weights = {
            'style_consistency': 0.25,
            'completeness': 0.35,
            'accuracy': 0.40
        }
        
        return (
            weights['style_consistency'] * self.style_consistency_score +
            weights['completeness'] * self.completeness_score +
            weights['accuracy'] * self.accuracy_score
        )