"""
Models for draft generation with citation integration.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

from citation_manager.models import Reference


class CitationStrategy(Enum):
    """Different strategies for integrating citations."""
    PROBLEM_GAP_SOLUTION = "problem_gap_solution"
    HYPOTHESIS_DRIVEN = "hypothesis_driven"
    METHODOLOGY_FOCUSED = "methodology_focused"
    LITERATURE_SYNTHESIS = "literature_synthesis"


class ReferenceRole(Enum):
    """Role of a reference in the introduction."""
    FOUNDATIONAL = "foundational"  # Basic concepts and established knowledge
    SUPPORTING = "supporting"      # Supporting evidence for claims
    CONTRASTING = "contrasting"    # Contrasting or alternative perspectives
    GAP_IDENTIFYING = "gap_identifying"  # Papers that identify research gaps
    METHODOLOGICAL = "methodological"    # Methodological approaches and techniques


@dataclass
class ReferenceContext:
    """Context information for how a reference should be used."""
    reference: Reference
    role: ReferenceRole
    key_concepts: List[str]
    key_findings: List[str]
    methodological_details: List[str]
    limitations_identified: List[str]
    citation_context: str  # Where/how it should be cited
    paragraph_target: int  # Which paragraph it should appear in (1-based)
    citation_format: str   # Exact citation format to use


@dataclass
class StudySpecification:
    """Specification of the study for which introduction is being written."""
    study_title: str
    research_type: str  # clinical_trial, experimental_study, etc.
    research_domain: str  # neurosurgery, cognitive_neuroscience, etc.
    primary_research_question: str
    primary_hypothesis: str
    study_objectives: List[str]
    target_population: str
    methodology_summary: str
    expected_outcomes: List[str]
    clinical_significance: str
    target_journal: Optional[str] = None
    ethical_considerations: Optional[str] = None


@dataclass
class CitationIntegrationPlan:
    """Plan for how citations will be integrated into the introduction."""
    strategy: CitationStrategy
    reference_contexts: List[ReferenceContext]
    paragraph_structure: Dict[int, str]  # paragraph number -> purpose
    citation_density_targets: Dict[int, float]  # paragraph -> target citations per sentence
    total_expected_citations: int
    critical_citations: List[str]  # Must-include citations
    
    def get_references_for_paragraph(self, paragraph_num: int) -> List[ReferenceContext]:
        """Get all references intended for a specific paragraph."""
        return [rc for rc in self.reference_contexts if rc.paragraph_target == paragraph_num]


@dataclass
class GeneratedDraft:
    """Generated introduction draft with citation integration."""
    draft_id: str
    study_specification: StudySpecification
    integration_plan: CitationIntegrationPlan
    generated_text: str
    paragraph_breakdown: List[str]
    citations_used: List[str]
    citations_missing: List[str]
    quality_scores: Dict[str, float]
    empirical_validation: Dict[str, Any]
    generated_at: datetime = field(default_factory=datetime.now)
    
    @property
    def citation_completeness(self) -> float:
        """Calculate what percentage of planned citations were included."""
        total_planned = len(self.integration_plan.reference_contexts)
        if total_planned == 0:
            return 1.0
        used = len(self.citations_used)
        return min(used / total_planned, 1.0)
    
    @property
    def word_count(self) -> int:
        """Get word count of generated text."""
        return len(self.generated_text.split())


@dataclass
class CitationCoherenceResult:
    """Result of citation coherence analysis."""
    overall_coherence: float  # 0-1 score
    paragraph_coherence: Dict[int, float]  # per-paragraph coherence
    citation_context_matches: Dict[str, float]  # citation -> context match score
    logical_flow_score: float
    citation_density_appropriateness: float
    issues_identified: List[str]
    recommendations: List[str]
    
    @property
    def is_publication_ready(self) -> bool:
        """Check if the introduction meets publication standards."""
        return (self.overall_coherence >= 0.8 and 
                self.logical_flow_score >= 0.8 and
                self.citation_density_appropriateness >= 0.7)


@dataclass
class ValidationReport:
    """Comprehensive validation report for generated draft."""
    draft_id: str
    empirical_pattern_compliance: float
    citation_coherence: CitationCoherenceResult
    factual_accuracy: float
    statistical_validity: float
    terminology_appropriateness: float
    overall_quality: float
    critical_issues: List[str]
    minor_issues: List[str]
    recommendations: List[str]
    ready_for_submission: bool
    validation_timestamp: datetime = field(default_factory=datetime.now)