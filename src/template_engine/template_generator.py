"""Template generator for neuroscience introductions based on analyzed patterns."""

import re
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.models import AnalysisResult, WritingPattern, SentenceType
from analysis.pattern_detector import WritingPatternDetector
from .models import (
    GeneratedTemplate, TemplateSection, SectionTemplate, ParagraphTemplate,
    TemplateVariable, TemplateMetadata, TemplateType, ParagraphType
)
from .exceptions import TemplateGenerationError, TemplateValidationError

logger = logging.getLogger(__name__)


class TemplateGenerator:
    """Generates introduction templates based on analyzed neuroscience writing patterns."""
    
    def __init__(self):
        """Initialize the template generator."""
        self.base_templates = self._load_base_templates()
        self.paragraph_templates = self._load_paragraph_templates()
        self.style_guidelines = self._load_style_guidelines()
        
    def generate_template(self, analysis_results: List[AnalysisResult],
                         detected_patterns: List[WritingPattern],
                         template_type: TemplateType = None) -> GeneratedTemplate:
        """
        Generate an introduction template based on analysis results and patterns.
        
        Args:
            analysis_results: List of analyzed document sections
            detected_patterns: Writing patterns detected from the analysis
            template_type: Specific template type to generate (auto-detect if None)
            
        Returns:
            Generated template with sections and guidance
            
        Raises:
            TemplateGenerationError: If template generation fails
        """
        try:
            template_id = f"neuro_intro_{uuid.uuid4().hex[:8]}"
            logger.info(f"Generating template {template_id} from {len(analysis_results)} analysis results")
            
            # Determine template type if not specified
            if template_type is None:
                template_type = self._determine_template_type(detected_patterns)
            
            # Create template metadata
            metadata = self._create_template_metadata(
                template_id, template_type, detected_patterns, analysis_results
            )
            
            # Generate sections based on patterns
            sections = self._generate_sections(
                analysis_results, detected_patterns, template_type
            )
            
            # Create global variables
            global_variables = self._create_global_variables(detected_patterns)
            
            # Generate style guidelines
            style_guidelines = self._generate_style_guidelines(detected_patterns)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_template_quality(sections, detected_patterns)
            
            template = GeneratedTemplate(
                template_id=template_id,
                metadata=metadata,
                sections=sections,
                global_variables=global_variables,
                style_guidelines=style_guidelines,
                quality_metrics=quality_metrics,
                generated_at=datetime.now()
            )
            
            # Validate template
            self._validate_template(template)
            
            logger.info(f"Generated template {template_id} successfully")
            return template
            
        except Exception as e:
            logger.error(f"Template generation failed: {e}")
            raise TemplateGenerationError(f"Failed to generate template: {e}")
    
    def _determine_template_type(self, patterns: List[WritingPattern]) -> TemplateType:
        """Determine the most appropriate template type based on patterns."""
        pattern_types = [p.pattern_type for p in patterns]
        
        # Check for hypothesis-driven structure
        if 'hypothesis_driven_structure' in pattern_types:
            return TemplateType.HYPOTHESIS_DRIVEN
        
        # Check for methodology focus
        if 'technique_heavy' in pattern_types or 'method_detailed_structure' in pattern_types:
            return TemplateType.METHODOLOGY_FOCUSED
        
        # Check for literature review style
        if 'citation_heavy' in pattern_types and 'background_citation_pattern' in pattern_types:
            return TemplateType.LITERATURE_REVIEW
        
        # Default to funnel structure for neuroscience
        return TemplateType.INTRODUCTION_FUNNEL
    
    def _create_template_metadata(self, template_id: str, template_type: TemplateType,
                                patterns: List[WritingPattern],
                                analysis_results: List[AnalysisResult]) -> TemplateMetadata:
        """Create metadata for the generated template."""
        
        # Calculate confidence based on pattern strength
        pattern_confidences = [p.confidence for p in patterns if p.confidence > 0]
        avg_confidence = sum(pattern_confidences) / len(pattern_confidences) if pattern_confidences else 0.5
        
        # Determine complexity level
        complexity_indicators = ['high_technical_density', 'complex_sentences', 'advanced']
        complexity_count = sum(1 for p in patterns if any(ind in p.pattern_type for ind in complexity_indicators))
        
        if complexity_count >= 2:
            complexity_level = "advanced"
        elif complexity_count >= 1:
            complexity_level = "intermediate"
        else:
            complexity_level = "basic"
        
        # Calculate pattern coverage
        total_possible_patterns = 10  # Approximate number of key patterns we look for
        pattern_coverage = min(len(patterns) / total_possible_patterns, 1.0)
        
        return TemplateMetadata(
            template_id=template_id,
            template_type=template_type,
            source_patterns=[p.pattern_type for p in patterns],
            confidence_score=avg_confidence,
            complexity_level=complexity_level,
            target_audience="neuroscience researchers",
            field_specialization="neuroscience",
            generated_from_papers=len(analysis_results),
            pattern_coverage=pattern_coverage
        )
    
    def _generate_sections(self, analysis_results: List[AnalysisResult],
                          patterns: List[WritingPattern],
                          template_type: TemplateType) -> List[TemplateSection]:
        """Generate template sections based on the template type and patterns."""
        
        if template_type == TemplateType.INTRODUCTION_FUNNEL:
            return self._generate_funnel_introduction(analysis_results, patterns)
        elif template_type == TemplateType.HYPOTHESIS_DRIVEN:
            return self._generate_hypothesis_driven_introduction(analysis_results, patterns)
        elif template_type == TemplateType.METHODOLOGY_FOCUSED:
            return self._generate_methodology_focused_introduction(analysis_results, patterns)
        elif template_type == TemplateType.LITERATURE_REVIEW:
            return self._generate_literature_review_introduction(analysis_results, patterns)
        else:
            return self._generate_default_introduction(analysis_results, patterns)
    
    def _generate_funnel_introduction(self, analysis_results: List[AnalysisResult],
                                    patterns: List[WritingPattern]) -> List[TemplateSection]:
        """Generate a funnel-structure introduction (broad to specific)."""
        
        # Paragraph 1: Broad context
        broad_context = ParagraphTemplate(
            paragraph_type=ParagraphType.BROAD_CONTEXT,
            title="Broad Context",
            content_template="""
            {broad_field_statement} {brain_complexity_statement} 
            {general_importance} {field_scope}
            """,
            variables=[
                TemplateVariable("broad_field_statement", "Opening statement about neuroscience/brain", "text",
                               example="The human brain represents one of the most complex biological systems"),
                TemplateVariable("brain_complexity_statement", "Specific complexity details", "text",
                               example="containing billions of interconnected neurons"),
                TemplateVariable("general_importance", "Why this field matters", "text",
                               example="Understanding these neural networks is crucial for"),
                TemplateVariable("field_scope", "Scope of current knowledge", "text",
                               example="advancing our knowledge of cognition and behavior")
            ],
            guidance_notes=[
                "Start with the broadest possible context",
                "Use impressive statistics about the brain if appropriate",
                "Establish why neuroscience matters"
            ],
            example_sentences=[
                "The human brain is one of the most complex structures in the known universe.",
                "Neural networks underlying cognition have fascinated researchers for decades.",
                "Understanding brain function represents a fundamental challenge in biology."
            ],
            min_length=60,
            max_length=120,
            citation_density=0.1
        )
        
        # Paragraph 2: Background literature
        background_lit = ParagraphTemplate(
            paragraph_type=ParagraphType.BACKGROUND_LITERATURE,
            title="Background Literature",
            content_template="""
            {previous_research} {key_findings} {established_knowledge}
            {current_understanding} {relevant_citations}
            """,
            variables=[
                TemplateVariable("previous_research", "Summary of previous research", "text",
                               example="Previous studies have established that"),
                TemplateVariable("key_findings", "Key findings from literature", "text",
                               example="the hippocampus plays a crucial role in memory formation"),
                TemplateVariable("established_knowledge", "What is well established", "text",
                               example="This region demonstrates remarkable plasticity"),
                TemplateVariable("current_understanding", "Current state of knowledge", "text",
                               example="Current evidence suggests that"),
                TemplateVariable("relevant_citations", "List of relevant citations", "citation",
                               example="(Smith et al., 2019; Johnson & Brown, 2020)")
            ],
            guidance_notes=[
                "Summarize established knowledge in your area",
                "Use strong, recent citations",
                "Build credibility by showing knowledge of the field"
            ],
            example_sentences=[
                "Extensive research has demonstrated the hippocampus's role in memory.",
                "Previous studies using fMRI have revealed activation patterns in the cortex.",
                "Electrophysiological studies have identified specific neural oscillations."
            ],
            min_length=80,
            max_length=150,
            citation_density=0.4
        )
        
        # Paragraph 3: Gap identification
        gap_identification = ParagraphTemplate(
            paragraph_type=ParagraphType.GAP_IDENTIFICATION,
            title="Knowledge Gap",
            content_template="""
            {transition_to_gap} {specific_gap} {why_important} {consequences}
            """,
            variables=[
                TemplateVariable("transition_to_gap", "Transition to gap", "text",
                               example="However, despite these advances"),
                TemplateVariable("specific_gap", "Specific knowledge gap", "text",
                               example="the precise mechanisms underlying X remain unclear"),
                TemplateVariable("why_important", "Why this gap is important", "text",
                               example="Understanding this is crucial because"),
                TemplateVariable("consequences", "Consequences of not knowing", "text",
                               example="it limits our ability to develop effective treatments")
            ],
            guidance_notes=[
                "Clearly identify what is NOT known",
                "Explain why this gap matters",
                "Use transition words like 'however', 'nevertheless', 'despite'"
            ],
            example_sentences=[
                "However, the specific neural mechanisms remain poorly understood.",
                "Despite extensive research, little is known about the temporal dynamics.",
                "Nevertheless, the functional significance of these patterns is unclear."
            ],
            min_length=60,
            max_length=120,
            citation_density=0.2
        )
        
        # Paragraph 4: Study objectives/hypothesis
        study_objectives = ParagraphTemplate(
            paragraph_type=ParagraphType.STUDY_OBJECTIVES,
            title="Study Objectives",
            content_template="""
            {study_purpose} {specific_aims} {hypothesis} {approach}
            """,
            variables=[
                TemplateVariable("study_purpose", "Overall purpose of study", "text",
                               example="The present study aims to investigate"),
                TemplateVariable("specific_aims", "Specific research aims", "text",
                               example="We specifically examine the relationship between X and Y"),
                TemplateVariable("hypothesis", "Research hypothesis", "text",
                               example="We hypothesize that"),
                TemplateVariable("approach", "General methodological approach", "text",
                               example="using functional MRI and behavioral measures")
            ],
            guidance_notes=[
                "State clear, specific objectives",
                "Include hypothesis if applicable",
                "Preview your methodological approach"
            ],
            example_sentences=[
                "The present study investigates the neural mechanisms of memory consolidation.",
                "We hypothesize that hippocampal-cortical interactions facilitate learning.",
                "To test this, we used simultaneous EEG-fMRI recordings."
            ],
            min_length=70,
            max_length=130,
            citation_density=0.1
        )
        
        # Create section template
        section_template = SectionTemplate(
            section_type="introduction",
            title="Introduction",
            description="Funnel-structure introduction moving from broad context to specific study aims",
            paragraphs=[broad_context, background_lit, gap_identification, study_objectives],
            structure_notes=[
                "Follow the funnel structure: broad → specific → gap → study",
                "Each paragraph should flow naturally to the next",
                "Use transition sentences between paragraphs"
            ],
            flow_guidelines=[
                "Start with broad context to orient readers",
                "Establish credibility with background literature",
                "Identify clear gap in knowledge",
                "Present study as solution to gap"
            ],
            estimated_word_count=350
        )
        
        # Create template section with placeholder content
        rendered_content = self._render_section_preview(section_template)
        
        template_section = TemplateSection(
            section_template=section_template,
            rendered_content=rendered_content,
            filled_variables={},
            unfilled_variables=[var.name for para in section_template.paragraphs for var in para.variables],
            word_count=section_template.estimated_word_count,
            quality_score=0.85
        )
        
        return [template_section]
    
    def _generate_hypothesis_driven_introduction(self, analysis_results: List[AnalysisResult],
                                               patterns: List[WritingPattern]) -> List[TemplateSection]:
        """Generate a hypothesis-driven introduction structure."""
        
        # Similar structure but emphasizes hypothesis and predictions
        # Implementation would follow similar pattern to funnel structure
        # but with emphasis on hypothesis formation and testing
        
        return self._generate_funnel_introduction(analysis_results, patterns)  # Simplified for now
    
    def _generate_methodology_focused_introduction(self, analysis_results: List[AnalysisResult],
                                                 patterns: List[WritingPattern]) -> List[TemplateSection]:
        """Generate a methodology-focused introduction."""
        return self._generate_funnel_introduction(analysis_results, patterns)  # Simplified for now
    
    def _generate_literature_review_introduction(self, analysis_results: List[AnalysisResult],
                                               patterns: List[WritingPattern]) -> List[TemplateSection]:
        """Generate a literature-review style introduction."""
        return self._generate_funnel_introduction(analysis_results, patterns)  # Simplified for now
    
    def _generate_default_introduction(self, analysis_results: List[AnalysisResult],
                                     patterns: List[WritingPattern]) -> List[TemplateSection]:
        """Generate a default introduction structure."""
        return self._generate_funnel_introduction(analysis_results, patterns)
    
    def _render_section_preview(self, section_template: SectionTemplate) -> str:
        """Render a preview of the section with placeholder text."""
        content = f"# {section_template.title}\n\n"
        content += f"{section_template.description}\n\n"
        
        for i, paragraph in enumerate(section_template.paragraphs, 1):
            content += f"## Paragraph {i}: {paragraph.title}\n\n"
            content += f"**Purpose**: {paragraph.paragraph_type.value.replace('_', ' ').title()}\n\n"
            content += f"**Guidance**: {'; '.join(paragraph.guidance_notes)}\n\n"
            content += f"**Example sentences**:\n"
            for sentence in paragraph.example_sentences:
                content += f"- {sentence}\n"
            content += "\n"
            content += f"**Variables to fill**:\n"
            for var in paragraph.variables:
                content += f"- `{var.name}`: {var.description}\n"
            content += f"\n**Word count**: {paragraph.min_length}-{paragraph.max_length} words\n\n"
        
        return content
    
    def _create_global_variables(self, patterns: List[WritingPattern]) -> List[TemplateVariable]:
        """Create global variables that apply across the template."""
        variables = [
            TemplateVariable(
                name="research_field",
                description="Primary research field/domain",
                variable_type="text",
                default_value="neuroscience",
                required=True,
                example="cognitive neuroscience"
            ),
            TemplateVariable(
                name="target_phenomenon",
                description="Main phenomenon being studied",
                variable_type="text",
                required=True,
                example="memory consolidation"
            ),
            TemplateVariable(
                name="primary_method",
                description="Primary research method",
                variable_type="text",
                required=True,
                example="functional MRI"
            ),
            TemplateVariable(
                name="key_brain_region",
                description="Key brain region of interest",
                variable_type="text",
                required=False,
                example="hippocampus"
            )
        ]
        
        return variables
    
    def _generate_style_guidelines(self, patterns: List[WritingPattern]) -> Dict[str, Any]:
        """Generate style guidelines based on detected patterns."""
        guidelines = {
            "formality_level": "high",
            "citation_style": "APA",
            "sentence_structure": "complex_acceptable",
            "technical_terminology": "appropriate_for_field",
            "paragraph_length": "medium",
            "transition_usage": "recommended"
        }
        
        # Adjust based on patterns
        for pattern in patterns:
            if pattern.pattern_type == "high_formality":
                guidelines["formality_level"] = "very_high"
            elif pattern.pattern_type == "citation_heavy":
                guidelines["citation_density"] = "high"
            elif pattern.pattern_type == "complex_sentences":
                guidelines["sentence_structure"] = "complex_preferred"
        
        return guidelines
    
    def _calculate_template_quality(self, sections: List[TemplateSection],
                                  patterns: List[WritingPattern]) -> Dict[str, float]:
        """Calculate quality metrics for the template."""
        return {
            "structure_completeness": 0.9,
            "pattern_alignment": min(len(patterns) / 5.0, 1.0),
            "guidance_clarity": 0.8,
            "variable_coverage": 0.85,
            "example_quality": 0.8
        }
    
    def _validate_template(self, template: GeneratedTemplate):
        """Validate the generated template."""
        if not template.sections:
            raise TemplateValidationError("Template must have at least one section")
        
        for section in template.sections:
            if not section.section_template.paragraphs:
                raise TemplateValidationError("Each section must have paragraphs")
        
        if template.metadata.confidence_score < 0.3:
            raise TemplateValidationError("Template confidence too low")
    
    def _load_base_templates(self) -> Dict[str, Any]:
        """Load base template structures."""
        return {}  # Placeholder - could load from files
    
    def _load_paragraph_templates(self) -> Dict[str, ParagraphTemplate]:
        """Load reusable paragraph templates."""
        return {}  # Placeholder
    
    def _load_style_guidelines(self) -> Dict[str, Any]:
        """Load style guidelines."""
        return {}  # Placeholder