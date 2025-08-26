"""Research project specification system for targeted introduction generation."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime


class StudyType(Enum):
    """Types of neuroscience studies."""
    CLINICAL_TRIAL = "clinical_trial"
    OBSERVATIONAL_STUDY = "observational_study"
    EXPERIMENTAL_STUDY = "experimental_study"
    META_ANALYSIS = "meta_analysis"
    SYSTEMATIC_REVIEW = "systematic_review"
    CASE_STUDY = "case_study"
    LONGITUDINAL_STUDY = "longitudinal_study"
    CROSS_SECTIONAL = "cross_sectional"
    PRECLINICAL_STUDY = "preclinical_study"


class ResearchDomain(Enum):
    """Specific research domains within neuroscience."""
    COGNITIVE_NEUROSCIENCE = "cognitive_neuroscience"
    CLINICAL_NEUROSCIENCE = "clinical_neuroscience"
    NEUROSURGERY = "neurosurgery"
    NEURO_ONCOLOGY = "neuro_oncology"
    NEUROPSYCHOLOGY = "neuropsychology"
    COMPUTATIONAL_NEUROSCIENCE = "computational_neuroscience"
    DEVELOPMENTAL_NEUROSCIENCE = "developmental_neuroscience"
    NEUROIMAGING = "neuroimaging"
    ELECTROPHYSIOLOGY = "electrophysiology"
    MOLECULAR_NEUROSCIENCE = "molecular_neuroscience"


class OutcomeType(Enum):
    """Types of study outcomes/endpoints."""
    PRIMARY_CLINICAL = "primary_clinical"
    SECONDARY_CLINICAL = "secondary_clinical"
    NEUROIMAGING = "neuroimaging"
    BEHAVIORAL_COGNITIVE = "behavioral_cognitive"
    ELECTROPHYSIOLOGICAL = "electrophysiological"
    MOLECULAR_BIOMARKER = "molecular_biomarker"
    QUALITY_OF_LIFE = "quality_of_life"
    SAFETY_TOLERABILITY = "safety_tolerability"
    EXPLORATORY = "exploratory"


@dataclass
class StudyEndpoint:
    """Specific study endpoint definition."""
    endpoint_name: str
    endpoint_type: OutcomeType
    description: str
    measurement_method: str
    expected_direction: str  # increase, decrease, no_change, bidirectional
    clinical_significance: str
    statistical_approach: Optional[str] = None
    timepoints: List[str] = None
    
    def __post_init__(self):
        if self.timepoints is None:
            self.timepoints = []
        if not self.endpoint_name.strip():
            raise ValueError("Endpoint name cannot be empty")


@dataclass
class ResearchHypothesis:
    """Structured research hypothesis."""
    hypothesis_statement: str
    hypothesis_type: str  # primary, secondary, exploratory
    directional: bool  # True if directional, False if non-directional
    statistical_test: Optional[str] = None
    effect_size_expected: Optional[float] = None
    biological_rationale: str = ""
    supporting_literature: List[str] = None
    
    def __post_init__(self):
        if self.supporting_literature is None:
            self.supporting_literature = []
        if not self.hypothesis_statement.strip():
            raise ValueError("Hypothesis statement cannot be empty")


@dataclass
class StudyPopulation:
    """Study population characteristics."""
    target_population: str
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    sample_size: Optional[int] = None
    age_range: Optional[str] = None
    sex_distribution: Optional[str] = None
    clinical_characteristics: List[str] = None
    recruitment_setting: Optional[str] = None
    
    def __post_init__(self):
        if self.clinical_characteristics is None:
            self.clinical_characteristics = []
        if not self.inclusion_criteria:
            raise ValueError("At least one inclusion criterion required")


@dataclass
class Intervention:
    """Study intervention details."""
    intervention_name: str
    intervention_type: str  # pharmacological, surgical, behavioral, device, etc.
    description: str
    dose_regimen: Optional[str] = None
    duration: Optional[str] = None
    control_condition: Optional[str] = None
    mechanism_of_action: Optional[str] = None
    safety_profile: Optional[str] = None
    
    def __post_init__(self):
        if not self.intervention_name.strip():
            raise ValueError("Intervention name cannot be empty")


@dataclass
class MethodologicalApproach:
    """Detailed methodological approach."""
    study_design: str
    primary_techniques: List[str]  # fMRI, EEG, behavioral testing, etc.
    data_acquisition_details: List[str]
    analysis_pipeline: List[str]
    statistical_methods: List[str]
    quality_control_measures: List[str]
    blinding_strategy: Optional[str] = None
    randomization_method: Optional[str] = None
    
    def __post_init__(self):
        if not self.primary_techniques:
            raise ValueError("At least one primary technique required")


@dataclass
class ClinicalSignificance:
    """Clinical significance and translational aspects."""
    clinical_problem: str
    current_limitations: str
    translational_potential: str
    clinical_applications: List[str]
    patient_impact: str
    healthcare_implications: Optional[str] = None
    
    def __post_init__(self):
        if not self.clinical_problem.strip():
            raise ValueError("Clinical problem description required")


@dataclass
class LiteratureContext:
    """Literature context and positioning."""
    research_gap: str
    key_controversies: List[str]
    theoretical_framework: str
    methodological_advances: List[str]
    conflicting_evidence: List[str] = None
    emerging_paradigms: List[str] = None
    
    def __post_init__(self):
        if self.conflicting_evidence is None:
            self.conflicting_evidence = []
        if self.emerging_paradigms is None:
            self.emerging_paradigms = []
        if not self.research_gap.strip():
            raise ValueError("Research gap description required")


@dataclass
class ComprehensiveResearchSpecification:
    """Complete research project specification for targeted introduction generation."""
    
    # Basic Study Information
    study_title: str
    study_type: StudyType
    research_domain: ResearchDomain
    
    # Research Framework
    research_objectives: List[str]
    primary_hypothesis: ResearchHypothesis
    secondary_hypotheses: List[ResearchHypothesis]
    study_endpoints: List[StudyEndpoint]
    
    # Study Details
    study_population: StudyPopulation
    intervention: Optional[Intervention]
    methodological_approach: MethodologicalApproach
    
    # Context and Significance
    clinical_significance: ClinicalSignificance
    literature_context: LiteratureContext
    
    # Timeline and Logistics
    study_duration: str
    expected_completion: Optional[str] = None
    funding_source: Optional[str] = None
    
    # Metadata
    specification_id: str = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.specification_id is None:
            self.specification_id = f"spec_{hash(self.study_title)}_{self.created_at.strftime('%Y%m%d')}"
        
        # Validation
        if not self.study_title.strip():
            raise ValueError("Study title cannot be empty")
        if not self.research_objectives:
            raise ValueError("At least one research objective required")
        if not self.study_endpoints:
            raise ValueError("At least one study endpoint required")
    
    @property
    def primary_endpoints(self) -> List[StudyEndpoint]:
        """Get primary endpoints only."""
        return [ep for ep in self.study_endpoints if ep.endpoint_type == OutcomeType.PRIMARY_CLINICAL]
    
    @property
    def secondary_endpoints(self) -> List[StudyEndpoint]:
        """Get secondary endpoints."""
        return [ep for ep in self.study_endpoints 
                if ep.endpoint_type in [OutcomeType.SECONDARY_CLINICAL, OutcomeType.NEUROIMAGING, 
                                      OutcomeType.BEHAVIORAL_COGNITIVE]]
    
    @property
    def has_intervention(self) -> bool:
        """Check if study has an intervention."""
        return self.intervention is not None
    
    @property
    def is_clinical_study(self) -> bool:
        """Check if this is a clinical study."""
        return self.study_type in [StudyType.CLINICAL_TRIAL, StudyType.OBSERVATIONAL_STUDY]
    
    @property
    def complexity_score(self) -> float:
        """Calculate study complexity score (0-1)."""
        complexity_factors = 0
        
        # Multiple endpoints increase complexity
        complexity_factors += min(len(self.study_endpoints) / 5, 0.3)
        
        # Multiple hypotheses increase complexity  
        hypothesis_count = 1 + len(self.secondary_hypotheses)
        complexity_factors += min(hypothesis_count / 5, 0.2)
        
        # Intervention studies are more complex
        if self.has_intervention:
            complexity_factors += 0.2
        
        # Clinical studies are more complex
        if self.is_clinical_study:
            complexity_factors += 0.2
        
        # Multiple techniques increase complexity
        technique_count = len(self.methodological_approach.primary_techniques)
        complexity_factors += min(technique_count / 5, 0.1)
        
        return min(complexity_factors, 1.0)
    
    def get_introduction_focus_areas(self) -> List[str]:
        """Determine key focus areas for introduction based on study specification."""
        focus_areas = []
        
        # Clinical problem and significance
        focus_areas.append("clinical_significance")
        
        # Research gap and literature context
        focus_areas.append("literature_gap")
        
        # Methodological approach if novel
        if len(self.methodological_approach.primary_techniques) > 2:
            focus_areas.append("methodological_innovation")
        
        # Population-specific considerations
        if self.study_population.clinical_characteristics:
            focus_areas.append("population_specificity")
        
        # Intervention rationale if applicable
        if self.has_intervention:
            focus_areas.append("intervention_rationale")
        
        # Translational potential
        if self.clinical_significance.translational_potential:
            focus_areas.append("translational_impact")
        
        return focus_areas
    
    def generate_introduction_outline(self) -> Dict[str, List[str]]:
        """Generate a tailored introduction outline based on study specification."""
        outline = {
            "paragraph_1_broad_context": [
                f"Establish broad context of {self.research_domain.value.replace('_', ' ')}",
                f"Introduce the clinical problem: {self.clinical_significance.clinical_problem}",
                "Set the scope and importance of the research area"
            ],
            
            "paragraph_2_literature_background": [
                "Review current understanding and established knowledge",
                f"Discuss theoretical framework: {self.literature_context.theoretical_framework}",
                "Present key findings from previous research"
            ],
            
            "paragraph_3_gap_and_limitations": [
                f"Identify the research gap: {self.literature_context.research_gap}",
                f"Discuss current limitations: {self.clinical_significance.current_limitations}",
                "Highlight controversies or conflicting evidence" if self.literature_context.key_controversies else None
            ],
            
            "paragraph_4_study_rationale": [
                f"Present study rationale and approach: {self.study_type.value.replace('_', ' ')}",
                f"Introduce methodological innovation" if len(self.methodological_approach.primary_techniques) > 2 else None,
                f"Explain intervention rationale" if self.has_intervention else None
            ],
            
            "paragraph_5_objectives_and_hypotheses": [
                "State primary research objective clearly",
                f"Present primary hypothesis: {self.primary_hypothesis.hypothesis_statement}",
                "Outline key endpoints and expected outcomes",
                f"Describe translational potential: {self.clinical_significance.translational_potential}"
            ]
        }
        
        # Clean up None values
        for key, items in outline.items():
            outline[key] = [item for item in items if item is not None]
        
        return outline
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "study_title": self.study_title,
            "study_type": self.study_type.value,
            "research_domain": self.research_domain.value,
            "research_objectives": self.research_objectives,
            "primary_hypothesis": {
                **self.primary_hypothesis.__dict__,
                "hypothesis_type": self.primary_hypothesis.hypothesis_type
            },
            "secondary_hypotheses": [
                {**h.__dict__, "hypothesis_type": h.hypothesis_type} 
                for h in self.secondary_hypotheses
            ],
            "study_endpoints": [
                {**ep.__dict__, "endpoint_type": ep.endpoint_type.value} 
                for ep in self.study_endpoints
            ],
            "study_population": self.study_population.__dict__,
            "intervention": self.intervention.__dict__ if self.intervention else None,
            "methodological_approach": self.methodological_approach.__dict__,
            "clinical_significance": self.clinical_significance.__dict__,
            "literature_context": self.literature_context.__dict__,
            "study_duration": self.study_duration,
            "expected_completion": self.expected_completion,
            "funding_source": self.funding_source,
            "specification_id": self.specification_id,
            "created_at": self.created_at.isoformat(),
            # Computed properties
            "primary_endpoints": [
                {**ep.__dict__, "endpoint_type": ep.endpoint_type.value} 
                for ep in self.primary_endpoints
            ],
            "secondary_endpoints": [
                {**ep.__dict__, "endpoint_type": ep.endpoint_type.value} 
                for ep in self.secondary_endpoints
            ],
            "has_intervention": self.has_intervention,
            "is_clinical_study": self.is_clinical_study,
            "complexity_score": self.complexity_score,
            "introduction_focus_areas": self.get_introduction_focus_areas(),
            "introduction_outline": self.generate_introduction_outline()
        }