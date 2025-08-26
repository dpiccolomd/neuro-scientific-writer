"""Data models for NLP analysis results."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from datetime import datetime


class SentenceType(Enum):
    """Types of sentences in scientific writing."""
    HYPOTHESIS = "hypothesis"
    OBJECTIVE = "objective"
    METHOD_DESCRIPTION = "method_description"
    RESULT_STATEMENT = "result_statement"
    CONCLUSION = "conclusion"
    CITATION_CONTEXT = "citation_context"
    BACKGROUND = "background"
    TRANSITION = "transition"
    UNKNOWN = "unknown"


class WritingStyle(Enum):
    """Scientific writing style characteristics."""
    FORMAL_ACADEMIC = "formal_academic"
    TECHNICAL_DESCRIPTIVE = "technical_descriptive"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    EXPLANATORY = "explanatory"
    ARGUMENTATIVE = "argumentative"


@dataclass
class NeuroTerm:
    """Represents a neuroscience terminology instance."""
    term: str
    category: str  # anatomy, function, pathology, technique, etc.
    context: str
    confidence: float
    position: Tuple[int, int]  # start, end character positions
    synonyms: List[str] = None
    definition: Optional[str] = None
    
    def __post_init__(self):
        if self.synonyms is None:
            self.synonyms = []
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not self.term.strip():
            raise ValueError("Term cannot be empty")


@dataclass
class SentenceStructure:
    """Analysis of sentence structure and complexity."""
    text: str
    sentence_type: SentenceType
    word_count: int
    complexity_score: float  # 0-1, higher = more complex
    passive_voice: bool
    citation_count: int
    technical_term_count: int
    readability_score: float
    grammatical_features: Dict[str, Any]
    
    def __post_init__(self):
        if not self.text.strip():
            raise ValueError("Sentence text cannot be empty")
        if self.word_count <= 0:
            self.word_count = len(self.text.split())


@dataclass
class WritingPattern:
    """Represents a detected writing pattern."""
    pattern_type: str
    description: str
    frequency: float  # 0-1, how often this pattern appears
    examples: List[str]
    confidence: float
    section_types: List[str]  # which sections this pattern appears in
    linguistic_features: Dict[str, Any]
    
    def __post_init__(self):
        if not 0.0 <= self.frequency <= 1.0:
            raise ValueError("Frequency must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class CoherenceMetrics:
    """Metrics for text coherence and flow."""
    lexical_cohesion: float
    semantic_similarity: float
    transition_quality: float
    logical_flow_score: float
    topic_consistency: float
    overall_coherence: float
    
    def __post_init__(self):
        """Calculate overall coherence if not provided."""
        if self.overall_coherence == 0.0:
            self.overall_coherence = (
                self.lexical_cohesion * 0.2 +
                self.semantic_similarity * 0.3 +
                self.transition_quality * 0.2 +
                self.logical_flow_score * 0.15 +
                self.topic_consistency * 0.15
            )


@dataclass
class StyleCharacteristics:
    """Analysis of writing style characteristics."""
    formality_level: float  # 0-1, higher = more formal
    technical_density: float  # ratio of technical terms to total words
    sentence_complexity: float  # average complexity score
    citation_density: float  # citations per sentence
    passive_voice_ratio: float
    avg_sentence_length: float
    vocabulary_richness: float  # type-token ratio
    writing_style: WritingStyle
    consistency_score: float
    
    def __post_init__(self):
        """Validate all ratios are between 0 and 1."""
        for field_name in ['formality_level', 'technical_density', 'sentence_complexity', 
                          'citation_density', 'passive_voice_ratio', 'vocabulary_richness', 
                          'consistency_score']:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0.0 and 1.0")


@dataclass
class TerminologyAnalysis:
    """Analysis of neuroscience terminology usage."""
    total_terms: int
    unique_terms: int
    term_categories: Dict[str, int]  # category -> count
    most_frequent_terms: List[NeuroTerm]
    terminology_density: float
    category_distribution: Dict[str, float]  # category -> percentage
    complexity_level: str  # basic, intermediate, advanced
    
    def __post_init__(self):
        if self.total_terms < 0 or self.unique_terms < 0:
            raise ValueError("Term counts cannot be negative")
        if not 0.0 <= self.terminology_density <= 1.0:
            raise ValueError("Terminology density must be between 0.0 and 1.0")


@dataclass
class AnalysisResult:
    """Complete analysis result for a document or section."""
    text_id: str
    source_type: str  # 'document', 'section', 'paragraph'
    sentences: List[SentenceStructure]
    neuro_terms: List[NeuroTerm]
    writing_patterns: List[WritingPattern]
    style_characteristics: StyleCharacteristics
    terminology_analysis: TerminologyAnalysis
    coherence_metrics: CoherenceMetrics
    overall_quality_score: float
    processing_time: float
    analyzer_version: str = "1.0.0"
    analyzed_at: datetime = None
    
    def __post_init__(self):
        if self.analyzed_at is None:
            self.analyzed_at = datetime.now()
        if not 0.0 <= self.overall_quality_score <= 1.0:
            raise ValueError("Quality score must be between 0.0 and 1.0")
    
    @property
    def total_sentences(self) -> int:
        """Get total number of sentences analyzed."""
        return len(self.sentences)
    
    @property
    def total_words(self) -> int:
        """Get total word count from all sentences."""
        return sum(s.word_count for s in self.sentences)
    
    @property
    def avg_sentence_complexity(self) -> float:
        """Get average sentence complexity score."""
        if not self.sentences:
            return 0.0
        return sum(s.complexity_score for s in self.sentences) / len(self.sentences)
    
    @property
    def citation_density(self) -> float:
        """Get citation density (citations per sentence)."""
        if not self.sentences:
            return 0.0
        total_citations = sum(s.citation_count for s in self.sentences)
        return total_citations / len(self.sentences)
    
    def get_sentences_by_type(self, sentence_type: SentenceType) -> List[SentenceStructure]:
        """Get all sentences of a specific type."""
        return [s for s in self.sentences if s.sentence_type == sentence_type]
    
    def get_terms_by_category(self, category: str) -> List[NeuroTerm]:
        """Get all terms from a specific category."""
        return [t for t in self.neuro_terms if t.category.lower() == category.lower()]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text_id": self.text_id,
            "source_type": self.source_type,
            "sentences": [
                {
                    **sentence.__dict__,
                    "sentence_type": sentence.sentence_type.value
                } 
                for sentence in self.sentences
            ],
            "neuro_terms": [term.__dict__ for term in self.neuro_terms],
            "writing_patterns": [
                {
                    **pattern.__dict__
                } 
                for pattern in self.writing_patterns
            ],
            "style_characteristics": {
                **self.style_characteristics.__dict__,
                "writing_style": self.style_characteristics.writing_style.value
            },
            "terminology_analysis": self.terminology_analysis.__dict__,
            "coherence_metrics": self.coherence_metrics.__dict__,
            "overall_quality_score": self.overall_quality_score,
            "processing_time": self.processing_time,
            "analyzer_version": self.analyzer_version,
            "analyzed_at": self.analyzed_at.isoformat(),
            # Computed properties
            "total_sentences": self.total_sentences,
            "total_words": self.total_words,
            "avg_sentence_complexity": self.avg_sentence_complexity,
            "citation_density": self.citation_density
        }