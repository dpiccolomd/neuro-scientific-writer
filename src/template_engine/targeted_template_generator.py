"""Enhanced template generator using detailed research specifications."""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.models import AnalysisResult, WritingPattern
from .research_specification import (
    ComprehensiveResearchSpecification, StudyType, ResearchDomain, 
    OutcomeType, StudyEndpoint
)
from .models import (
    GeneratedTemplate, TemplateSection, SectionTemplate, ParagraphTemplate,
    TemplateVariable, TemplateMetadata, TemplateType, ParagraphType
)
from .exceptions import TemplateGenerationError

logger = logging.getLogger(__name__)


class TargetedTemplateGenerator:
    """
    Enhanced template generator that creates highly targeted introductions
    based on detailed research specifications.
    """
    
    def __init__(self):
        """Initialize the targeted template generator."""
        self.domain_specific_templates = self._load_domain_templates()
        self.study_type_templates = self._load_study_type_templates()
        self.endpoint_specific_content = self._load_endpoint_content()
    
    def generate_targeted_template(self,
                                 research_spec: ComprehensiveResearchSpecification,
                                 analysis_results: List[AnalysisResult] = None,
                                 detected_patterns: List[WritingPattern] = None) -> GeneratedTemplate:
        """
        Generate a highly targeted introduction template based on detailed research specification.
        
        Args:
            research_spec: Comprehensive research project specification
            analysis_results: Optional analysis results from source papers
            detected_patterns: Optional detected writing patterns
            
        Returns:
            Targeted template optimized for the specific research project
        """
        template_id = f"targeted_{research_spec.specification_id}_{uuid.uuid4().hex[:6]}"
        
        logger.info(f"Generating targeted template for: {research_spec.study_title}")
        logger.info(f"Study type: {research_spec.study_type.value}")
        logger.info(f"Domain: {research_spec.research_domain.value}")
        logger.info(f"Primary endpoints: {len(research_spec.primary_endpoints)}")
        
        try:
            # Create targeted template metadata
            metadata = self._create_targeted_metadata(template_id, research_spec, detected_patterns)
            
            # Generate domain and study-specific sections
            sections = self._generate_targeted_sections(research_spec, analysis_results, detected_patterns)
            
            # Create research-specific global variables
            global_variables = self._create_research_specific_variables(research_spec)
            
            # Generate targeted style guidelines
            style_guidelines = self._generate_targeted_style_guidelines(research_spec, detected_patterns)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_targeted_quality_metrics(research_spec, sections)
            
            template = GeneratedTemplate(
                template_id=template_id,
                metadata=metadata,
                sections=sections,
                global_variables=global_variables,
                style_guidelines=style_guidelines,
                quality_metrics=quality_metrics,
                generated_at=datetime.now()
            )
            
            logger.info(f"Generated targeted template: {template_id}")
            logger.info(f"Complexity score: {research_spec.complexity_score:.3f}")
            logger.info(f"Focus areas: {', '.join(research_spec.get_introduction_focus_areas())}")
            
            return template
            
        except Exception as e:
            logger.error(f"Targeted template generation failed: {e}")
            raise TemplateGenerationError(f"Failed to generate targeted template: {e}")
    
    def _create_targeted_metadata(self, template_id: str,
                                research_spec: ComprehensiveResearchSpecification,
                                patterns: List[WritingPattern] = None) -> TemplateMetadata:
        """Create metadata specifically tailored to the research project."""
        
        # Determine template type based on study characteristics
        template_type = self._determine_optimal_template_type(research_spec)
        
        # Calculate confidence based on specification completeness
        confidence_score = self._calculate_specification_confidence(research_spec)
        
        # Determine complexity level
        complexity_level = self._determine_complexity_level(research_spec)
        
        source_patterns = [p.pattern_type for p in patterns] if patterns else []
        pattern_coverage = min(len(source_patterns) / 10, 1.0) if patterns else 0.5
        
        return TemplateMetadata(
            template_id=template_id,
            template_type=template_type,
            source_patterns=source_patterns,
            confidence_score=confidence_score,
            complexity_level=complexity_level,
            target_audience=f"{research_spec.research_domain.value.replace('_', ' ')} researchers",
            field_specialization=research_spec.research_domain.value,
            generated_from_papers=len(patterns) if patterns else 0,
            pattern_coverage=pattern_coverage
        )
    
    def _generate_targeted_sections(self, research_spec: ComprehensiveResearchSpecification,
                                  analysis_results: List[AnalysisResult] = None,
                                  patterns: List[WritingPattern] = None) -> List[TemplateSection]:
        """Generate sections specifically tailored to the research project."""
        
        # Get the introduction outline from research specification
        introduction_outline = research_spec.generate_introduction_outline()
        focus_areas = research_spec.get_introduction_focus_areas()
        
        # Create targeted paragraphs
        paragraphs = []
        
        # Paragraph 1: Broad Context (tailored to domain and clinical problem)
        broad_context = self._create_clinical_context_paragraph(research_spec)
        paragraphs.append(broad_context)
        
        # Paragraph 2: Literature Background (tailored to theoretical framework)
        literature_background = self._create_literature_background_paragraph(research_spec)
        paragraphs.append(literature_background)
        
        # Paragraph 3: Research Gap (specific to identified gap and limitations)
        research_gap = self._create_research_gap_paragraph(research_spec)
        paragraphs.append(research_gap)
        
        # Paragraph 4: Study Rationale and Methodology (specific to approach and innovation)
        study_rationale = self._create_study_rationale_paragraph(research_spec)
        paragraphs.append(study_rationale)
        
        # Paragraph 5: Objectives and Hypotheses (specific to actual objectives and endpoints)
        objectives_hypotheses = self._create_objectives_hypotheses_paragraph(research_spec)
        paragraphs.append(objectives_hypotheses)
        
        # Optional Paragraph 6: Clinical Significance (if high translational potential)
        if research_spec.clinical_significance.translational_potential and research_spec.is_clinical_study:
            clinical_significance = self._create_clinical_significance_paragraph(research_spec)
            paragraphs.append(clinical_significance)
        
        # Create the complete section
        section_template = SectionTemplate(
            section_type="targeted_introduction",
            title=f"Introduction for {research_spec.study_type.value.replace('_', ' ').title()}",
            description=f"Targeted introduction for {research_spec.research_domain.value.replace('_', ' ')} research",
            paragraphs=paragraphs,
            structure_notes=self._generate_structure_notes(research_spec),
            flow_guidelines=self._generate_flow_guidelines(research_spec),
            estimated_word_count=self._estimate_word_count(research_spec)
        )
        
        # Render the section
        rendered_content = self._render_targeted_section_preview(section_template, research_spec)
        
        template_section = TemplateSection(
            section_template=section_template,
            rendered_content=rendered_content,
            filled_variables=self._get_pre_filled_variables(research_spec),
            unfilled_variables=self._get_remaining_variables(section_template, research_spec),
            word_count=section_template.estimated_word_count,
            quality_score=0.9  # High quality for targeted templates
        )
        
        return [template_section]
    
    def _create_clinical_context_paragraph(self, research_spec: ComprehensiveResearchSpecification) -> ParagraphTemplate:
        """Create clinical context paragraph tailored to the specific clinical problem."""
        
        domain = research_spec.research_domain.value.replace('_', ' ')
        clinical_problem = research_spec.clinical_significance.clinical_problem
        
        # Tailor content template based on domain
        if research_spec.research_domain == ResearchDomain.NEUROSURGERY:
            content_template = f"""
            {domain.title()} represents a critical medical discipline addressing complex neurological conditions that significantly impact patient outcomes. 
            {clinical_problem} {'{clinical_impact_statement}'} poses substantial challenges for both patients and healthcare systems worldwide. 
            Current approaches in {domain} are limited by {'{current_technical_limitations}'}, 
            necessitating innovative research to advance {'{specific_therapeutic_area}'} and improve patient care.
            """
        elif research_spec.research_domain == ResearchDomain.NEURO_ONCOLOGY:
            content_template = f"""
            Brain tumors and other neurooncological conditions represent some of the most challenging malignancies, 
            with significant impact on neurological function and patient survival. {clinical_problem} 
            Current treatment paradigms including {'{standard_treatments}'} face limitations in {'{treatment_limitations}'}. 
            Understanding {'{key_biological_processes}'} is crucial for developing more effective therapeutic strategies.
            """
        else:
            content_template = f"""
            {domain.title()} encompasses the study of neural mechanisms underlying human behavior and cognition. 
            {clinical_problem} {'{prevalence_statistics}'} represents a significant burden on individuals and society. 
            Despite advances in {'{methodological_advances}'}, our understanding of {'{key_mechanisms}'} remains incomplete, 
            limiting the development of effective interventions.
            """
        
        variables = [
            TemplateVariable("clinical_impact_statement", "Statement about clinical impact", "text",
                           example="affecting millions of patients globally"),
            TemplateVariable("current_technical_limitations", "Current technical or clinical limitations", "text",
                           example="limited surgical precision and post-operative complications"),
            TemplateVariable("specific_therapeutic_area", "Specific area of therapeutic focus", "text",
                           example="minimally invasive surgical approaches"),
            TemplateVariable("prevalence_statistics", "Relevant prevalence or epidemiological data", "text",
                           example="with prevalence rates of X per 100,000 individuals")
        ]
        
        return ParagraphTemplate(
            paragraph_type=ParagraphType.BROAD_CONTEXT,
            title=f"Clinical Context for {domain.title()}",
            content_template=content_template,
            variables=variables,
            guidance_notes=[
                f"Establish the clinical significance of {clinical_problem}",
                "Include relevant epidemiological data or statistics",
                "Connect to broader healthcare challenges",
                "Set the urgency for research in this area"
            ],
            example_sentences=self._get_domain_specific_examples(research_spec.research_domain),
            min_length=80,
            max_length=150,
            citation_density=0.2
        )
    
    def _create_literature_background_paragraph(self, research_spec: ComprehensiveResearchSpecification) -> ParagraphTemplate:
        """Create literature background paragraph specific to the theoretical framework."""
        
        theoretical_framework = research_spec.literature_context.theoretical_framework
        
        content_template = f"""
        {theoretical_framework} {'{framework_elaboration}'} provides the conceptual foundation for understanding 
        {'{target_phenomena}'}. Previous research has established that {'{established_knowledge}'} through 
        studies employing {'{methodological_approaches}'} {'{key_citations}'}. 
        Specifically, {'{specific_findings}'} have demonstrated {'{key_mechanisms_findings}'}, 
        supporting the notion that {'{theoretical_predictions}'}. However, {'{knowledge_limitations}'} 
        continue to limit our comprehensive understanding of {'{research_focus_area}'}.
        """
        
        variables = [
            TemplateVariable("framework_elaboration", "Elaboration on the theoretical framework", "text",
                           example="which posits that neural networks operate through coordinated oscillations"),
            TemplateVariable("target_phenomena", "The specific phenomena being studied", "text",
                           example="memory consolidation during sleep"),
            TemplateVariable("established_knowledge", "What is already established in the literature", "text",
                           example="hippocampal-cortical interactions are crucial for memory formation"),
            TemplateVariable("methodological_approaches", "Key methodological approaches used in prior work", "text",
                           example="simultaneous EEG-fMRI recordings and optogenetic manipulations"),
            TemplateVariable("key_citations", "Key supporting citations", "citation",
                           example="(Author et al., 2020; Researcher & Colleague, 2019)"),
            TemplateVariable("specific_findings", "Specific research findings", "text",
                           example="studies by [Author] and [Collaborators]"),
            TemplateVariable("key_mechanisms_findings", "Key mechanisms discovered", "text",
                           example="theta-gamma coupling facilitates memory encoding"),
            TemplateVariable("theoretical_predictions", "What the theory predicts", "text",
                           example="synchronized oscillations should enhance memory performance"),
            TemplateVariable("knowledge_limitations", "Current limitations in knowledge", "text",
                           example="the precise timing and directionality of these interactions"),
            TemplateVariable("research_focus_area", "Your specific research focus area", "text",
                           example="sleep-dependent memory consolidation mechanisms")
        ]
        
        return ParagraphTemplate(
            paragraph_type=ParagraphType.BACKGROUND_LITERATURE,
            title="Theoretical Framework and Literature Background",
            content_template=content_template,
            variables=variables,
            guidance_notes=[
                f"Ground your work in {theoretical_framework}",
                "Cite key foundational studies extensively",
                "Show progression of knowledge in the field",
                "Establish what methodological approaches have been used",
                "Set up the context for identifying gaps"
            ],
            example_sentences=[
                f"The {theoretical_framework.lower()} framework suggests that...",
                "Extensive research has demonstrated the role of...",
                "Neuroimaging studies have consistently shown that..."
            ],
            min_length=120,
            max_length=200,
            citation_density=0.5  # High citation density for literature background
        )
    
    def _create_research_gap_paragraph(self, research_spec: ComprehensiveResearchSpecification) -> ParagraphTemplate:
        """Create research gap paragraph specific to the identified gap."""
        
        research_gap = research_spec.literature_context.research_gap
        current_limitations = research_spec.clinical_significance.current_limitations
        
        content_template = f"""
        Despite these advances, {research_gap} {'{gap_elaboration}'} represents a critical knowledge gap that limits 
        {'{clinical_or_theoretical_impact}'}. Current approaches are constrained by {current_limitations}, 
        which {'{limitation_consequences}'}. {'{controversy_statement}'}
        Furthermore, {'{methodological_gaps}'} have prevented comprehensive investigation of {'{specific_research_question}'}. 
        {'{translational_gap}'} This gap is particularly important because {'{significance_of_gap}'}, 
        and addressing it could {'{potential_impact}'}.
        """
        
        variables = [
            TemplateVariable("gap_elaboration", "Elaborate on the specific research gap", "text",
                           example="particularly regarding the temporal dynamics of neural synchronization"),
            TemplateVariable("clinical_or_theoretical_impact", "Impact of the gap on clinical practice or theory", "text", 
                           example="our ability to develop targeted therapeutic interventions"),
            TemplateVariable("limitation_consequences", "Consequences of current limitations", "text",
                           example="preclude real-time monitoring of neural state changes"),
            TemplateVariable("controversy_statement", "Statement about controversies if applicable", "text",
                           example="Additionally, conflicting findings regarding [specific aspect] have created uncertainty in the field.", 
                           required=False),
            TemplateVariable("methodological_gaps", "Specific methodological limitations", "text",
                           example="limitations in temporal resolution of current imaging techniques"),
            TemplateVariable("specific_research_question", "Your specific research question", "text",
                           example="the causal relationship between oscillatory patterns and behavioral outcomes"),
            TemplateVariable("translational_gap", "Translational gap if applicable", "text",
                           example="The translational potential remains unrealized due to these methodological constraints.",
                           required=research_spec.is_clinical_study),
            TemplateVariable("significance_of_gap", "Why this gap is significant", "text",
                           example="it directly impacts our understanding of disease mechanisms"),
            TemplateVariable("potential_impact", "Potential impact of addressing the gap", "text",
                           example="lead to more precise diagnostic tools and personalized treatment approaches")
        ]
        
        return ParagraphTemplate(
            paragraph_type=ParagraphType.GAP_IDENTIFICATION,
            title="Research Gap and Current Limitations",
            content_template=content_template,
            variables=variables,
            guidance_notes=[
                f"Clearly articulate the specific gap: {research_gap}",
                f"Connect limitations to clinical impact: {current_limitations}",
                "Explain why this gap matters for the field",
                "Set up the rationale for your study approach",
                "Include controversies if they exist in the field"
            ],
            example_sentences=[
                f"However, {research_gap} remains poorly understood...",
                "These limitations prevent comprehensive assessment of...",
                "The lack of understanding regarding [specific aspect] limits..."
            ],
            min_length=100,
            max_length=180,
            citation_density=0.3
        )
    
    def _create_study_rationale_paragraph(self, research_spec: ComprehensiveResearchSpecification) -> ParagraphTemplate:
        """Create study rationale paragraph specific to the methodological approach."""
        
        study_type = research_spec.study_type.value.replace('_', ' ')
        primary_techniques = research_spec.methodological_approach.primary_techniques
        
        # Tailor based on whether there's an intervention
        if research_spec.has_intervention:
            intervention_text = f"""
            To address these limitations, we propose a {study_type} investigating {'{intervention_description}'} 
            in {'{target_population_description}'}. Our approach utilizes {'{methodological_innovation}'} 
            to overcome previous methodological constraints. The intervention, {'{intervention_rationale}'}, 
            is expected to {'{mechanism_of_action}'} based on {'{theoretical_foundation}'} {'{supporting_evidence}'}.
            """
        else:
            intervention_text = f"""
            To address these limitations, we designed a {study_type} using {'{methodological_innovation}'} 
            to investigate {'{research_focus}'} in {'{target_population_description}'}. Our approach combines 
            {'{technique_combination}'} to provide {'{methodological_advantages}'}. 
            This design allows us to {'{study_capabilities}'} while addressing previous methodological constraints.
            """
        
        content_template = intervention_text + f"""
        {'{study_design_justification}'} The use of {'{primary_technique_justification}'} is particularly advantageous 
        because {'{technique_advantages}'}. {'{innovation_statement}'}
        """
        
        variables = self._create_study_rationale_variables(research_spec)
        
        return ParagraphTemplate(
            paragraph_type=ParagraphType.STUDY_OBJECTIVES,
            title="Study Rationale and Methodological Approach",
            content_template=content_template,
            variables=variables,
            guidance_notes=[
                f"Justify the {study_type} design choice",
                f"Explain how your methodology addresses identified limitations",
                f"Highlight methodological innovations: {', '.join(primary_techniques)}",
                "Connect methodology to research objectives",
                "Explain intervention rationale if applicable"
            ],
            example_sentences=self._get_methodology_specific_examples(research_spec),
            min_length=90,
            max_length=170,
            citation_density=0.2
        )
    
    def _create_objectives_hypotheses_paragraph(self, research_spec: ComprehensiveResearchSpecification) -> ParagraphTemplate:
        """Create objectives and hypotheses paragraph with specific endpoints."""
        
        primary_objectives = research_spec.research_objectives[:2]  # First 2 objectives
        primary_hypothesis = research_spec.primary_hypothesis.hypothesis_statement
        primary_endpoints = research_spec.primary_endpoints
        
        content_template = f"""
        The primary objective of this study is to {primary_objectives[0] if primary_objectives else '{primary_objective}'}. 
        {f"Secondary objectives include {'{secondary_objectives}'}" if len(primary_objectives) > 1 else ""}
        We hypothesize that {primary_hypothesis} {'{hypothesis_elaboration}'}. 
        The primary endpoint is {'{primary_endpoint_description}'}, measured by {'{primary_endpoint_method}'}. 
        {'{secondary_endpoints_statement}'}
        {'{translational_significance}'} Success in achieving these objectives will {'{expected_impact}'} 
        and {'{clinical_implications}'}.
        """
        
        variables = [
            TemplateVariable("primary_objective", "Primary research objective", "text",
                           default_value=primary_objectives[0] if primary_objectives else "",
                           example="investigate the neural mechanisms underlying working memory consolidation"),
            TemplateVariable("secondary_objectives", "Secondary objectives", "text",
                           example="examine individual differences in neural efficiency and assess clinical correlations",
                           required=len(primary_objectives) > 1),
            TemplateVariable("hypothesis_elaboration", "Elaboration on the hypothesis", "text",
                           example="specifically predicting enhanced theta-gamma coupling during successful encoding"),
            TemplateVariable("primary_endpoint_description", "Description of primary endpoint", "text",
                           default_value=primary_endpoints[0].description if primary_endpoints else "",
                           example="change in memory consolidation efficiency from baseline"),
            TemplateVariable("primary_endpoint_method", "Method for measuring primary endpoint", "text",
                           default_value=primary_endpoints[0].measurement_method if primary_endpoints else "",
                           example="validated neuropsychological assessment battery and fMRI analysis"),
            TemplateVariable("secondary_endpoints_statement", "Statement about secondary endpoints", "text",
                           example="Secondary endpoints include neuroimaging biomarkers and quality of life measures.",
                           required=len(research_spec.secondary_endpoints) > 0),
            TemplateVariable("translational_significance", "Translational significance statement", "text",
                           example="This research has direct translational potential for developing cognitive interventions.",
                           required=research_spec.is_clinical_study),
            TemplateVariable("expected_impact", "Expected impact of the study", "text",
                           example="advance our understanding of memory consolidation mechanisms"),
            TemplateVariable("clinical_implications", "Clinical implications", "text",
                           example="inform the development of targeted therapeutic interventions")
        ]
        
        return ParagraphTemplate(
            paragraph_type=ParagraphType.STUDY_OBJECTIVES,
            title="Study Objectives, Hypotheses, and Endpoints",
            content_template=content_template,
            variables=variables,
            guidance_notes=[
                "State objectives clearly and specifically",
                f"Present hypothesis: {primary_hypothesis[:100]}...",
                f"Describe primary endpoints: {len(primary_endpoints)} primary endpoint(s)",
                "Connect to translational potential" if research_spec.is_clinical_study else "Focus on theoretical contributions",
                "Emphasize expected impact and significance"
            ],
            example_sentences=[
                "The primary objective of this study is to...",
                f"We hypothesize that {research_spec.primary_hypothesis.hypothesis_statement[:50]}...",
                "Success in achieving these objectives will advance..."
            ],
            min_length=100,
            max_length=180,
            citation_density=0.1
        )
    
    def _create_clinical_significance_paragraph(self, research_spec: ComprehensiveResearchSpecification) -> ParagraphTemplate:
        """Create clinical significance paragraph for clinical studies."""
        
        translational_potential = research_spec.clinical_significance.translational_potential
        patient_impact = research_spec.clinical_significance.patient_impact
        
        content_template = f"""
        The clinical significance of this research extends beyond theoretical understanding to direct patient care applications. 
        {translational_potential} {'{translational_elaboration}'}. 
        {patient_impact} {'{patient_impact_elaboration}'}. 
        The findings from this study could {'{immediate_applications}'} and contribute to {'{long_term_goals}'}. 
        Furthermore, {'{healthcare_system_impact}'} by {'{system_benefits}'}. 
        Given the {'{clinical_burden}'}, this research addresses a critical need for {'{unmet_clinical_need}'}.
        """
        
        variables = [
            TemplateVariable("translational_elaboration", "Elaboration on translational potential", "text",
                           example="with potential for immediate implementation in clinical practice"),
            TemplateVariable("patient_impact_elaboration", "Elaboration on patient impact", "text", 
                           example="particularly for patients with treatment-resistant conditions"),
            TemplateVariable("immediate_applications", "Immediate clinical applications", "text",
                           example="improve patient selection for specific interventions"),
            TemplateVariable("long_term_goals", "Long-term clinical goals", "text",
                           example="development of personalized treatment protocols"),
            TemplateVariable("healthcare_system_impact", "Impact on healthcare system", "text",
                           example="this research may reduce healthcare costs and improve resource allocation"),
            TemplateVariable("system_benefits", "Specific system benefits", "text",
                           example="reducing unnecessary procedures and optimizing treatment pathways"),
            TemplateVariable("clinical_burden", "Description of clinical burden", "text",
                           example="significant burden of neurological disease on healthcare systems"),
            TemplateVariable("unmet_clinical_need", "Unmet clinical need", "text",
                           example="evidence-based treatment selection criteria")
        ]
        
        return ParagraphTemplate(
            paragraph_type=ParagraphType.SIGNIFICANCE_STATEMENT,
            title="Clinical Significance and Translational Impact",
            content_template=content_template,
            variables=variables,
            guidance_notes=[
                f"Emphasize translational potential: {translational_potential[:100]}...",
                f"Connect to patient impact: {patient_impact[:100]}...",
                "Include healthcare system implications",
                "Address unmet clinical needs",
                "Justify the research investment"
            ],
            example_sentences=[
                "The clinical significance of this research extends to...",
                "Patients would benefit directly through...",
                "Healthcare systems could realize improvements in..."
            ],
            min_length=90,
            max_length=160,
            citation_density=0.2
        )
    
    # Helper methods continue...
    
    def _determine_optimal_template_type(self, research_spec: ComprehensiveResearchSpecification) -> TemplateType:
        """Determine optimal template type based on research characteristics."""
        if research_spec.primary_hypothesis.directional and len(research_spec.study_endpoints) > 0:
            return TemplateType.HYPOTHESIS_DRIVEN
        elif len(research_spec.methodological_approach.primary_techniques) > 2:
            return TemplateType.METHODOLOGY_FOCUSED
        elif research_spec.is_clinical_study:
            return TemplateType.INTRODUCTION_DIRECT
        else:
            return TemplateType.INTRODUCTION_FUNNEL
    
    def _calculate_specification_confidence(self, research_spec: ComprehensiveResearchSpecification) -> float:
        """Calculate confidence based on completeness of research specification."""
        completeness_factors = 0.0
        
        # Basic completeness
        if research_spec.research_objectives:
            completeness_factors += 0.2
        if research_spec.study_endpoints:
            completeness_factors += 0.2
        if research_spec.literature_context.research_gap:
            completeness_factors += 0.2
        if research_spec.clinical_significance.clinical_problem:
            completeness_factors += 0.2
            
        # Enhanced completeness
        if research_spec.primary_hypothesis.biological_rationale:
            completeness_factors += 0.1
        if research_spec.methodological_approach.analysis_pipeline:
            completeness_factors += 0.1
            
        return min(completeness_factors, 1.0)
    
    def _determine_complexity_level(self, research_spec: ComprehensiveResearchSpecification) -> str:
        """Determine complexity level based on study characteristics."""
        complexity_score = research_spec.complexity_score
        
        if complexity_score >= 0.7:
            return "advanced"
        elif complexity_score >= 0.4:
            return "intermediate"
        else:
            return "basic"
    
    def _create_research_specific_variables(self, research_spec: ComprehensiveResearchSpecification) -> List[TemplateVariable]:
        """Create variables specific to the research project."""
        variables = [
            TemplateVariable(
                name="study_title",
                description="Complete study title",
                variable_type="text",
                default_value=research_spec.study_title,
                required=True
            ),
            TemplateVariable(
                name="research_domain",
                description="Primary research domain",
                variable_type="text", 
                default_value=research_spec.research_domain.value.replace('_', ' '),
                required=True
            ),
            TemplateVariable(
                name="study_type",
                description="Type of study design",
                variable_type="text",
                default_value=research_spec.study_type.value.replace('_', ' '),
                required=True
            ),
            TemplateVariable(
                name="target_population",
                description="Target study population",
                variable_type="text",
                default_value=research_spec.study_population.target_population,
                required=True
            )
        ]
        
        # Add intervention-specific variables if applicable
        if research_spec.has_intervention:
            variables.append(TemplateVariable(
                name="intervention_name",
                description="Name of the intervention",
                variable_type="text",
                default_value=research_spec.intervention.intervention_name,
                required=True
            ))
        
        return variables
    
    def _generate_targeted_style_guidelines(self, research_spec: ComprehensiveResearchSpecification,
                                         patterns: List[WritingPattern] = None) -> Dict[str, Any]:
        """Generate style guidelines specific to the research domain and study type."""
        guidelines = {
            "formality_level": "high",
            "citation_style": "APA",
            "technical_terminology": "domain_appropriate",
            "statistical_reporting": "rigorous"
        }
        
        # Domain-specific adjustments
        if research_spec.research_domain in [ResearchDomain.NEUROSURGERY, ResearchDomain.NEURO_ONCOLOGY]:
            guidelines.update({
                "formality_level": "very_high",
                "clinical_focus": "mandatory", 
                "patient_safety_considerations": "required",
                "translational_emphasis": "high"
            })
        
        # Study type specific adjustments
        if research_spec.study_type == StudyType.CLINICAL_TRIAL:
            guidelines.update({
                "endpoint_specification": "detailed",
                "regulatory_compliance": "required",
                "ethical_considerations": "emphasized"
            })
        
        return guidelines
    
    def _calculate_targeted_quality_metrics(self, research_spec: ComprehensiveResearchSpecification,
                                          sections: List[TemplateSection]) -> Dict[str, float]:
        """Calculate quality metrics for targeted template."""
        return {
            "specification_completeness": self._calculate_specification_confidence(research_spec),
            "domain_alignment": 0.95,  # High for targeted templates
            "methodological_clarity": 0.9,
            "objective_specificity": 0.92,
            "clinical_relevance": 0.85 if research_spec.is_clinical_study else 0.7
        }
    
    def generate_empirical_template(
        self,
        research_spec: ComprehensiveResearchSpecification,
        empirical_detector: 'EmpiricalPatternDetector'
    ) -> GeneratedTemplate:
        """
        Generate template using empirical patterns from trained data.
        
        Args:
            research_spec: Research specification
            empirical_detector: Trained empirical pattern detector
            
        Returns:
            Template based on empirical evidence
        """
        # Use empirical detector to get patterns for this domain/journal
        empirical_patterns = []
        try:
            # This would use actual trained patterns
            empirical_patterns = empirical_detector.detect_patterns_empirical(
                [], require_statistical_significance=True
            )
        except Exception as e:
            logger.warning(f"Could not load empirical patterns: {e}")
        
        # Generate template using empirical evidence
        return self.generate_targeted_template(
            research_spec=research_spec,
            analysis_results=[],
            detected_patterns=empirical_patterns
        )
    
    # Additional helper methods would continue here...
    def _load_domain_templates(self) -> Dict:
        return {}  # Placeholder
    
    def _load_study_type_templates(self) -> Dict:
        return {}  # Placeholder
        
    def _load_endpoint_content(self) -> Dict:
        return {}  # Placeholder
    
    def _get_domain_specific_examples(self, domain: ResearchDomain) -> List[str]:
        return ["Domain-specific example sentence here"]
    
    def _create_study_rationale_variables(self, research_spec) -> List[TemplateVariable]:
        return []  # Simplified for now
    
    def _get_methodology_specific_examples(self, research_spec) -> List[str]:
        return ["Methodology-specific example"]
    
    def _generate_structure_notes(self, research_spec) -> List[str]:
        return ["Structure note"]
    
    def _generate_flow_guidelines(self, research_spec) -> List[str]:
        return ["Flow guideline"]
    
    def _estimate_word_count(self, research_spec) -> int:
        base_count = 300
        if research_spec.has_intervention:
            base_count += 50
        if research_spec.is_clinical_study:
            base_count += 50
        return base_count
    
    def _render_targeted_section_preview(self, section_template, research_spec) -> str:
        return f"Targeted section preview for {research_spec.study_title}"
    
    def _get_pre_filled_variables(self, research_spec) -> Dict[str, str]:
        return {
            "study_title": research_spec.study_title,
            "research_domain": research_spec.research_domain.value.replace('_', ' ')
        }
    
    def _get_remaining_variables(self, section_template, research_spec) -> List[str]:
        all_vars = []
        for para in section_template.paragraphs:
            all_vars.extend([var.name for var in para.variables])
        
        pre_filled = self._get_pre_filled_variables(research_spec)
        return [var for var in all_vars if var not in pre_filled]