"""Data models for quality control and validation."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from datetime import datetime


class WarningLevel(Enum):
    """Severity levels for validation warnings."""
    INFO = "info"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationType(Enum):
    """Types of validation checks."""
    CITATION_VERIFICATION = "citation_verification"
    FACTUAL_CONSISTENCY = "factual_consistency"
    STATISTICAL_ACCURACY = "statistical_accuracy"
    TERMINOLOGY_ACCURACY = "terminology_accuracy"
    PLAGIARISM_CHECK = "plagiarism_check"
    METHODOLOGICAL_RIGOR = "methodological_rigor"
    NUMERICAL_VALIDATION = "numerical_validation"
    REFERENCE_CURRENCY = "reference_currency"


@dataclass
class ValidationIssue:
    """Represents a validation issue or concern."""
    issue_type: ValidationType
    severity: WarningLevel
    message: str
    location: str  # Where in the text the issue occurs
    suggestion: Optional[str] = None
    confidence: float = 1.0  # How confident we are in this issue (0-1)
    source_reference: Optional[str] = None  # Reference to source material
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not self.message.strip():
            raise ValueError("Issue message cannot be empty")


@dataclass
class CitationValidation:
    """Results of citation verification."""
    citation_text: str
    found_in_sources: bool
    context_appropriate: bool
    formatting_correct: bool
    source_accessible: bool
    confidence_score: float
    issues: List[ValidationIssue]
    
    def __post_init__(self):
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")


@dataclass
class FactualClaim:
    """Represents a factual claim that needs verification."""
    claim_text: str
    claim_type: str  # anatomical, functional, statistical, methodological
    location: str
    confidence: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    verification_status: str  # verified, uncertain, contradicted, unknown
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.verification_status not in ['verified', 'uncertain', 'contradicted', 'unknown']:
            raise ValueError("Invalid verification status")


@dataclass
class StatisticalCheck:
    """Results of statistical validation."""
    statistic_text: str
    statistic_type: str  # p-value, correlation, effect size, etc.
    value_reported: Optional[float]
    value_range_valid: bool
    format_correct: bool
    context_appropriate: bool
    issues: List[ValidationIssue]


@dataclass
class TerminologyValidation:
    """Results of neuroscience terminology validation."""
    term: str
    category: str  # anatomy, function, pathology, technique
    definition_correct: bool
    usage_appropriate: bool
    spelling_correct: bool
    alternative_terms: List[str]
    confidence_score: float
    
    def __post_init__(self):
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")


@dataclass
class PlagiarismResult:
    """Results of plagiarism detection."""
    text_segment: str
    similarity_score: float  # 0-1, higher means more similar
    potential_sources: List[str]
    is_problematic: bool
    severity: WarningLevel
    
    def __post_init__(self):
        if not 0.0 <= self.similarity_score <= 1.0:
            raise ValueError("Similarity score must be between 0.0 and 1.0")


@dataclass
class ValidationResult:
    """Results from a specific validation check."""
    validation_type: ValidationType
    passed: bool
    score: float  # 0-1, higher is better
    issues: List[ValidationIssue]
    details: Dict[str, Any]
    processing_time: float
    
    def __post_init__(self):
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        return any(issue.severity == WarningLevel.CRITICAL for issue in self.issues)
    
    @property
    def has_high_severity_issues(self) -> bool:
        """Check if there are any high or critical severity issues."""
        return any(issue.severity in [WarningLevel.HIGH, WarningLevel.CRITICAL] 
                  for issue in self.issues)


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics."""
    overall_score: float
    citation_accuracy: float
    factual_consistency: float
    statistical_validity: float
    terminology_score: float
    plagiarism_score: float
    methodological_rigor: float
    confidence_interval: Tuple[float, float]  # (lower, upper) bounds
    
    def __post_init__(self):
        """Validate all scores are between 0 and 1."""
        scores = [
            self.overall_score, self.citation_accuracy, self.factual_consistency,
            self.statistical_validity, self.terminology_score, self.plagiarism_score,
            self.methodological_rigor
        ]
        for score in scores:
            if not 0.0 <= score <= 1.0:
                raise ValueError("All scores must be between 0.0 and 1.0")
        
        # Validate confidence interval
        if not (0.0 <= self.confidence_interval[0] <= self.confidence_interval[1] <= 1.0):
            raise ValueError("Invalid confidence interval")


@dataclass
class QualityReport:
    """Comprehensive quality validation report."""
    document_id: str
    validation_timestamp: datetime
    quality_metrics: QualityMetrics
    validation_results: List[ValidationResult]
    critical_issues: List[ValidationIssue]
    warnings: List[ValidationIssue]
    recommendations: List[str]
    sources_validated: int
    validation_coverage: float  # Percentage of content validated
    ready_for_publication: bool
    
    def __post_init__(self):
        if self.validation_timestamp is None:
            self.validation_timestamp = datetime.now()
        if not 0.0 <= self.validation_coverage <= 1.0:
            raise ValueError("Validation coverage must be between 0.0 and 1.0")
    
    @property
    def total_issues(self) -> int:
        """Get total number of issues across all validations."""
        return len(self.critical_issues) + len(self.warnings)
    
    @property
    def critical_issue_count(self) -> int:
        """Get number of critical issues."""
        return len([issue for issue in self.critical_issues 
                   if issue.severity == WarningLevel.CRITICAL])
    
    @property
    def high_severity_issue_count(self) -> int:
        """Get number of high severity issues."""
        return len([issue for issue in self.critical_issues + self.warnings
                   if issue.severity == WarningLevel.HIGH])
    
    @property
    def publication_readiness_score(self) -> float:
        """Calculate publication readiness score (0-1)."""
        if self.critical_issue_count > 0:
            return 0.0
        
        # Weighted score based on quality metrics
        weights = {
            'overall': 0.3,
            'citation': 0.2,
            'factual': 0.2,
            'statistical': 0.1,
            'terminology': 0.1,
            'plagiarism': 0.1
        }
        
        score = (
            weights['overall'] * self.quality_metrics.overall_score +
            weights['citation'] * self.quality_metrics.citation_accuracy +
            weights['factual'] * self.quality_metrics.factual_consistency +
            weights['statistical'] * self.quality_metrics.statistical_validity +
            weights['terminology'] * self.quality_metrics.terminology_score +
            weights['plagiarism'] * self.quality_metrics.plagiarism_score
        )
        
        # Penalize for high severity issues
        high_severity_penalty = self.high_severity_issue_count * 0.1
        return max(0.0, score - high_severity_penalty)
    
    def get_issues_by_severity(self, severity: WarningLevel) -> List[ValidationIssue]:
        """Get all issues of a specific severity level."""
        all_issues = self.critical_issues + self.warnings
        return [issue for issue in all_issues if issue.severity == severity]
    
    def get_issues_by_type(self, validation_type: ValidationType) -> List[ValidationIssue]:
        """Get all issues of a specific validation type."""
        all_issues = self.critical_issues + self.warnings
        return [issue for issue in all_issues if issue.issue_type == validation_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "document_id": self.document_id,
            "validation_timestamp": self.validation_timestamp.isoformat(),
            "quality_metrics": {
                **self.quality_metrics.__dict__,
                "confidence_interval": list(self.quality_metrics.confidence_interval)
            },
            "validation_results": [
                {
                    **result.__dict__,
                    "validation_type": result.validation_type.value,
                    "issues": [
                        {
                            **issue.__dict__,
                            "issue_type": issue.issue_type.value,
                            "severity": issue.severity.value
                        } 
                        for issue in result.issues
                    ]
                }
                for result in self.validation_results
            ],
            "critical_issues": [
                {
                    **issue.__dict__,
                    "issue_type": issue.issue_type.value,
                    "severity": issue.severity.value
                }
                for issue in self.critical_issues
            ],
            "warnings": [
                {
                    **issue.__dict__,
                    "issue_type": issue.issue_type.value,
                    "severity": issue.severity.value
                }
                for issue in self.warnings
            ],
            "recommendations": self.recommendations,
            "sources_validated": self.sources_validated,
            "validation_coverage": self.validation_coverage,
            "ready_for_publication": self.ready_for_publication,
            # Computed properties
            "total_issues": self.total_issues,
            "critical_issue_count": self.critical_issue_count,
            "high_severity_issue_count": self.high_severity_issue_count,
            "publication_readiness_score": self.publication_readiness_score
        }