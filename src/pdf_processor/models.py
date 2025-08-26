"""Data models for PDF processing."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class SectionType(Enum):
    """Types of document sections."""
    TITLE = "title"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    METHODS = "methods"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    ACKNOWLEDGMENTS = "acknowledgments"
    UNKNOWN = "unknown"


@dataclass
class Citation:
    """Represents a citation extracted from the document."""
    text: str
    authors: List[str]
    title: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    page_number: Optional[int] = None
    confidence: float = 0.0
    
    def __post_init__(self):
        """Validate citation data after initialization."""
        if not self.text.strip():
            raise ValueError("Citation text cannot be empty")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class DocumentSection:
    """Represents a section of the document."""
    section_type: SectionType
    title: str
    content: str
    page_start: int
    page_end: int
    citations: List[Citation]
    word_count: int
    confidence: float = 0.0
    
    def __post_init__(self):
        """Validate section data after initialization."""
        if not self.content.strip():
            raise ValueError("Section content cannot be empty")
        if self.page_start < 1 or self.page_end < self.page_start:
            raise ValueError("Invalid page numbers")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        # Calculate word count if not provided
        if self.word_count <= 0:
            self.word_count = len(self.content.split())


@dataclass
class DocumentMetadata:
    """Metadata extracted from the document."""
    title: Optional[str] = None
    authors: List[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    keywords: List[str] = None
    
    def __post_init__(self):
        """Initialize empty lists if None."""
        if self.authors is None:
            self.authors = []
        if self.keywords is None:
            self.keywords = []


@dataclass
class ProcessingStats:
    """Statistics about the processing operation."""
    total_pages: int
    total_words: int
    total_citations: int
    sections_detected: int
    processing_time: float
    extraction_method: str
    confidence_score: float
    warnings: List[str]
    
    def __post_init__(self):
        """Initialize warnings list if None."""
        if self.warnings is None:
            self.warnings = []


@dataclass
class ProcessedDocument:
    """Complete processed document with all extracted information."""
    file_path: str
    metadata: DocumentMetadata
    sections: List[DocumentSection]
    full_text: str
    stats: ProcessingStats
    processed_at: datetime
    processor_version: str = "1.0.0"
    
    def __post_init__(self):
        """Validate processed document data."""
        if not self.file_path:
            raise ValueError("File path is required")
        if not self.sections:
            raise ValueError("Document must have at least one section")
        if not self.full_text.strip():
            raise ValueError("Full text cannot be empty")
    
    @property
    def introduction_section(self) -> Optional[DocumentSection]:
        """Get the introduction section if it exists."""
        for section in self.sections:
            if section.section_type == SectionType.INTRODUCTION:
                return section
        return None
    
    @property
    def abstract_section(self) -> Optional[DocumentSection]:
        """Get the abstract section if it exists."""
        for section in self.sections:
            if section.section_type == SectionType.ABSTRACT:
                return section
        return None
    
    @property
    def all_citations(self) -> List[Citation]:
        """Get all citations from all sections."""
        citations = []
        for section in self.sections:
            citations.extend(section.citations)
        return citations
    
    def get_sections_by_type(self, section_type: SectionType) -> List[DocumentSection]:
        """Get all sections of a specific type."""
        return [section for section in self.sections if section.section_type == section_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "file_path": self.file_path,
            "metadata": self.metadata.__dict__,
            "sections": [
                {
                    **section.__dict__,
                    "section_type": section.section_type.value,
                    "citations": [citation.__dict__ for citation in section.citations]
                }
                for section in self.sections
            ],
            "full_text": self.full_text,
            "stats": {
                **self.stats.__dict__,
                "warnings": self.stats.warnings
            },
            "processed_at": self.processed_at.isoformat(),
            "processor_version": self.processor_version
        }