"""
Citation-Aware Introduction Generator

Generates introduction drafts that integrate specific reference papers
with empirically-trained writing patterns to create publication-ready
scientific introductions.
"""

import logging
import uuid
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .models import (
    StudySpecification, CitationIntegrationPlan, ReferenceContext, 
    GeneratedDraft, CitationStrategy, ReferenceRole
)
from citation_manager.models import Reference
from citation_manager.apa_formatter import APAFormatter
from template_engine import TargetedTemplateGenerator, ResearchSpecification
from template_engine.research_specification import StudyType
from analysis.empirical_pattern_detector import EmpiricalPatternDetector

logger = logging.getLogger(__name__)


class CitationAwareGenerator:
    """
    Advanced generator that creates introductions with integrated citations
    based on empirical patterns and specific reference requirements.
    """
    
    def __init__(self, empirical_patterns_dir: str):
        """Initialize citation-aware generator with empirical patterns."""
        self.empirical_detector = EmpiricalPatternDetector(empirical_patterns_dir)
        self.template_generator = TargetedTemplateGenerator()
        self.apa_formatter = APAFormatter()
        
    def generate_citation_aware_introduction(
        self,
        study_spec: StudySpecification,
        reference_contexts: List[ReferenceContext],
        citation_strategy: CitationStrategy = CitationStrategy.PROBLEM_GAP_SOLUTION,
        target_word_count: int = 400
    ) -> GeneratedDraft:
        """
        Generate introduction draft with integrated citations.
        
        Args:
            study_spec: Complete study specification
            reference_contexts: List of references with usage contexts
            citation_strategy: Strategy for citation integration
            target_word_count: Target word count for introduction
            
        Returns:
            Generated draft with citation integration
        """
        draft_id = f"draft_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Generating citation-aware introduction: {draft_id}")
        logger.info(f"Study: {study_spec.study_title}")
        logger.info(f"References: {len(reference_contexts)}")
        logger.info(f"Strategy: {citation_strategy.value}")
        
        try:
            # Create citation integration plan
            integration_plan = self._create_citation_integration_plan(
                study_spec, reference_contexts, citation_strategy
            )
            
            # Generate empirical template structure
            research_spec = self._convert_to_research_specification(study_spec)
            base_template = self.template_generator.generate_empirical_template(
                research_spec, self.empirical_detector
            )
            
            # Generate citation-integrated draft
            generated_text = self._generate_text_with_citations(
                study_spec, integration_plan, base_template, target_word_count
            )
            
            # Extract paragraph breakdown
            paragraphs = self._split_into_paragraphs(generated_text)
            
            # Identify citations used and missing
            citations_used = self._extract_citations_from_text(generated_text)
            citations_missing = self._identify_missing_citations(
                integration_plan, citations_used
            )
            
            # Calculate quality scores
            quality_scores = self._calculate_quality_scores(
                generated_text, integration_plan, study_spec
            )
            
            # Get empirical validation
            empirical_validation = self._get_empirical_validation(
                generated_text, study_spec.research_domain
            )
            
            draft = GeneratedDraft(
                draft_id=draft_id,
                study_specification=study_spec,
                integration_plan=integration_plan,
                generated_text=generated_text,
                paragraph_breakdown=paragraphs,
                citations_used=citations_used,
                citations_missing=citations_missing,
                quality_scores=quality_scores,
                empirical_validation=empirical_validation
            )
            
            logger.info(f"Generated draft: {len(generated_text.split())} words")
            logger.info(f"Citations used: {len(citations_used)}/{len(reference_contexts)}")
            logger.info(f"Quality score: {quality_scores.get('overall', 0.0):.3f}")
            
            return draft
            
        except Exception as e:
            logger.error(f"Draft generation failed: {e}")
            raise
    
    def _create_citation_integration_plan(
        self,
        study_spec: StudySpecification,
        reference_contexts: List[ReferenceContext],
        strategy: CitationStrategy
    ) -> CitationIntegrationPlan:
        """Create plan for integrating citations into introduction structure."""
        
        # Define paragraph structure based on strategy
        if strategy == CitationStrategy.PROBLEM_GAP_SOLUTION:
            paragraph_structure = {
                1: "Broad context and clinical significance",
                2: "Current knowledge and established findings", 
                3: "Research gap and limitations",
                4: "Study objectives and approach"
            }
            citation_density_targets = {1: 0.3, 2: 0.6, 3: 0.4, 4: 0.2}
        elif strategy == CitationStrategy.HYPOTHESIS_DRIVEN:
            paragraph_structure = {
                1: "Theoretical framework and background",
                2: "Previous findings and evidence",
                3: "Hypothesis development and rationale", 
                4: "Study design and predictions"
            }
            citation_density_targets = {1: 0.4, 2: 0.7, 3: 0.5, 4: 0.2}
        else:  # Default structure
            paragraph_structure = {
                1: "Introduction to research area",
                2: "Literature review and current state", 
                3: "Knowledge gaps and study rationale",
                4: "Objectives and methodology"
            }
            citation_density_targets = {1: 0.3, 2: 0.6, 3: 0.4, 4: 0.2}
        
        # Assign references to paragraphs based on their roles
        self._assign_references_to_paragraphs(reference_contexts, strategy)
        
        # Identify critical citations
        critical_citations = [
            rc.citation_format for rc in reference_contexts 
            if rc.role in [ReferenceRole.FOUNDATIONAL, ReferenceRole.GAP_IDENTIFYING]
        ]
        
        return CitationIntegrationPlan(
            strategy=strategy,
            reference_contexts=reference_contexts,
            paragraph_structure=paragraph_structure,
            citation_density_targets=citation_density_targets,
            total_expected_citations=len(reference_contexts),
            critical_citations=critical_citations
        )
    
    def _assign_references_to_paragraphs(
        self, 
        reference_contexts: List[ReferenceContext],
        strategy: CitationStrategy
    ):
        """Assign references to appropriate paragraphs based on their roles."""
        
        role_to_paragraph = {
            CitationStrategy.PROBLEM_GAP_SOLUTION: {
                ReferenceRole.FOUNDATIONAL: 1,
                ReferenceRole.SUPPORTING: 2,
                ReferenceRole.CONTRASTING: 2,
                ReferenceRole.GAP_IDENTIFYING: 3,
                ReferenceRole.METHODOLOGICAL: 4
            },
            CitationStrategy.HYPOTHESIS_DRIVEN: {
                ReferenceRole.FOUNDATIONAL: 1,
                ReferenceRole.SUPPORTING: 2,
                ReferenceRole.CONTRASTING: 2,
                ReferenceRole.GAP_IDENTIFYING: 3,
                ReferenceRole.METHODOLOGICAL: 4
            }
        }
        
        mapping = role_to_paragraph.get(strategy, role_to_paragraph[CitationStrategy.PROBLEM_GAP_SOLUTION])
        
        for ref_context in reference_contexts:
            if ref_context.paragraph_target == 0:  # Not yet assigned
                ref_context.paragraph_target = mapping.get(ref_context.role, 2)
    
    def _convert_to_research_specification(self, study_spec: StudySpecification) -> ResearchSpecification:
        """Convert StudySpecification to ResearchSpecification for template generation."""
        
        # Map research types
        type_mapping = {
            'clinical_trial': StudyType.CLINICAL_TRIAL,
            'experimental_study': StudyType.EXPERIMENTAL_STUDY,
            'observational_study': StudyType.OBSERVATIONAL_STUDY,
            'systematic_review': StudyType.SYSTEMATIC_REVIEW,
            'case_study': StudyType.CASE_STUDY
        }
        
        study_type = type_mapping.get(study_spec.research_type, StudyType.EXPERIMENTAL_STUDY)
        
        return ResearchSpecification(
            study_title=study_spec.study_title,
            study_type=study_type,
            research_domain=study_spec.research_domain,
            target_journal=study_spec.target_journal
        )
    
    def _generate_text_with_citations(
        self,
        study_spec: StudySpecification,
        integration_plan: CitationIntegrationPlan,
        base_template,
        target_word_count: int
    ) -> str:
        """Generate the actual introduction text with integrated citations."""
        
        paragraphs = []
        
        for para_num in sorted(integration_plan.paragraph_structure.keys()):
            paragraph_purpose = integration_plan.paragraph_structure[para_num]
            paragraph_refs = integration_plan.get_references_for_paragraph(para_num)
            
            paragraph_text = self._generate_paragraph_with_citations(
                para_num, paragraph_purpose, paragraph_refs, study_spec, integration_plan
            )
            paragraphs.append(paragraph_text)
        
        # Join paragraphs and adjust length to target
        full_text = "\n\n".join(paragraphs)
        full_text = self._adjust_text_length(full_text, target_word_count)
        
        return full_text
    
    def _generate_paragraph_with_citations(
        self,
        para_num: int,
        purpose: str, 
        references: List[ReferenceContext],
        study_spec: StudySpecification,
        integration_plan: CitationIntegrationPlan
    ) -> str:
        """Generate a single paragraph with appropriate citations."""
        
        # Base content templates by paragraph purpose
        if para_num == 1:  # Broad context
            base_text = self._generate_context_paragraph(study_spec, references)
        elif para_num == 2:  # Literature review
            base_text = self._generate_literature_paragraph(study_spec, references)
        elif para_num == 3:  # Research gap
            base_text = self._generate_gap_paragraph(study_spec, references)
        elif para_num == 4:  # Study objectives
            base_text = self._generate_objectives_paragraph(study_spec, references)
        else:
            base_text = f"This paragraph focuses on {purpose}."
        
        # Integrate specific citations
        final_text = self._integrate_citations_into_paragraph(base_text, references)
        
        return final_text
    
    def _generate_context_paragraph(self, study_spec: StudySpecification, references: List[ReferenceContext]) -> str:
        """Generate broad context paragraph."""
        domain_context = {
            'neurosurgery': f"Neurosurgical interventions represent a critical component of modern medical care, "
                          f"addressing complex neurological conditions that significantly impact patient outcomes. ",
            'cognitive_neuroscience': f"Cognitive neuroscience has revealed fundamental principles governing "
                                    f"human brain function and behavior through decades of systematic research. ",
            'neuroimaging': f"Advanced neuroimaging techniques have revolutionized our understanding of brain "
                           f"structure and function, providing unprecedented insights into neural mechanisms. "
        }
        
        base_context = domain_context.get(
            study_spec.research_domain, 
            f"Research in {study_spec.research_domain.replace('_', ' ')} has established important "
            f"foundations for understanding complex neurobiological processes. "
        )
        
        # Add clinical significance
        clinical_context = (f"{study_spec.clinical_significance} This represents a significant "
                          f"challenge for both patients and healthcare systems worldwide. ")
        
        return base_context + clinical_context
    
    def _generate_literature_paragraph(self, study_spec: StudySpecification, references: List[ReferenceContext]) -> str:
        """Generate literature review paragraph with supporting evidence."""
        lit_intro = ("Previous research has established several key findings that form the foundation "
                    "for the current investigation. ")
        
        # Add methodology context if references include methodological papers
        method_refs = [r for r in references if r.role == ReferenceRole.METHODOLOGICAL]
        if method_refs:
            method_context = ("Recent methodological advances have enabled more precise investigation "
                            "of these mechanisms, providing new insights into underlying processes. ")
            lit_intro += method_context
        
        return lit_intro
    
    def _generate_gap_paragraph(self, study_spec: StudySpecification, references: List[ReferenceContext]) -> str:
        """Generate research gap paragraph."""
        gap_intro = "Despite these advances, several important questions remain unanswered. "
        
        # Extract limitations from gap-identifying references
        gap_refs = [r for r in references if r.role == ReferenceRole.GAP_IDENTIFYING]
        if gap_refs:
            limitations_text = ("Current approaches are limited by methodological constraints and "
                              "incomplete understanding of underlying mechanisms. ")
            gap_intro += limitations_text
        
        # Connect to study rationale
        rationale = (f"Addressing these limitations is crucial for advancing our understanding "
                    f"of {study_spec.research_domain.replace('_', ' ')} and improving clinical outcomes. ")
        
        return gap_intro + rationale
    
    def _generate_objectives_paragraph(self, study_spec: StudySpecification, references: List[ReferenceContext]) -> str:
        """Generate study objectives paragraph."""
        objective_intro = f"The primary objective of this study is to {study_spec.primary_research_question} "
        
        hypothesis = f"We hypothesize that {study_spec.primary_hypothesis} "
        
        methodology = f"This {study_spec.research_type.replace('_', ' ')} employs {study_spec.methodology_summary} "
        
        expected_impact = ("The findings from this investigation will contribute to advancing theoretical "
                          "understanding and may have important clinical implications. ")
        
        return objective_intro + hypothesis + methodology + expected_impact
    
    def _integrate_citations_into_paragraph(self, paragraph_text: str, references: List[ReferenceContext]) -> str:
        """Integrate specific citations into paragraph text."""
        if not references:
            return paragraph_text
        
        sentences = paragraph_text.split('. ')
        final_sentences = []
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            # Add appropriate citations based on sentence content and reference contexts
            sentence_with_citations = sentence.strip()
            
            # Find relevant citations for this sentence
            relevant_citations = self._find_relevant_citations_for_sentence(sentence, references)
            
            if relevant_citations:
                citation_text = self._format_multiple_citations(relevant_citations)
                sentence_with_citations += f" {citation_text}"
            
            if sentence_with_citations and not sentence_with_citations.endswith('.'):
                sentence_with_citations += '.'
                
            final_sentences.append(sentence_with_citations)
        
        return ' '.join(final_sentences)
    
    def _find_relevant_citations_for_sentence(self, sentence: str, references: List[ReferenceContext]) -> List[str]:
        """Find citations relevant to a specific sentence based on content matching."""
        relevant = []
        sentence_lower = sentence.lower()
        
        for ref_context in references:
            # Check if sentence mentions concepts from this reference
            concept_matches = any(
                concept.lower() in sentence_lower 
                for concept in ref_context.key_concepts
            )
            
            # Check if sentence mentions findings from this reference
            finding_matches = any(
                finding.lower() in sentence_lower
                for finding in ref_context.key_findings
            )
            
            if concept_matches or finding_matches:
                relevant.append(ref_context.citation_format)
        
        return relevant[:3]  # Limit to 3 citations per sentence
    
    def _format_multiple_citations(self, citations: List[str]) -> str:
        """Format multiple citations according to APA style."""
        if len(citations) == 1:
            return f"({citations[0]})"
        elif len(citations) == 2:
            return f"({citations[0]}; {citations[1]})"
        else:
            citation_list = "; ".join(citations)
            return f"({citation_list})"
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into individual paragraphs."""
        return [p.strip() for p in text.split('\n\n') if p.strip()]
    
    def _extract_citations_from_text(self, text: str) -> List[str]:
        """Extract all citations from generated text."""
        import re
        citation_pattern = r'\(([^)]+)\)'
        matches = re.findall(citation_pattern, text)
        
        # Filter to only actual citations (contain author names and years)
        citations = []
        for match in matches:
            if any(char.isdigit() for char in match) and len(match.split()) >= 2:
                # Split multiple citations within same parentheses
                individual_citations = [c.strip() for c in match.split(';')]
                citations.extend(individual_citations)
        
        return list(set(citations))  # Remove duplicates
    
    def _identify_missing_citations(
        self, 
        integration_plan: CitationIntegrationPlan, 
        citations_used: List[str]
    ) -> List[str]:
        """Identify citations that were planned but not included in the text."""
        planned_citations = set()
        for ref_context in integration_plan.reference_contexts:
            # Extract author and year from citation format
            citation_key = ref_context.citation_format.replace(',', '').replace('.', '')
            planned_citations.add(citation_key)
        
        used_citations = set()
        for citation in citations_used:
            clean_citation = citation.replace(',', '').replace('.', '')
            used_citations.add(clean_citation)
        
        missing = []
        for planned in planned_citations:
            if not any(planned.lower() in used.lower() for used in used_citations):
                missing.append(planned)
        
        return missing
    
    def _calculate_quality_scores(
        self, 
        generated_text: str, 
        integration_plan: CitationIntegrationPlan,
        study_spec: StudySpecification
    ) -> Dict[str, float]:
        """Calculate quality scores for the generated draft."""
        scores = {}
        
        # Citation completeness
        total_planned = len(integration_plan.reference_contexts)
        citations_used = len(self._extract_citations_from_text(generated_text))
        scores['citation_completeness'] = min(citations_used / max(total_planned, 1), 1.0)
        
        # Text length appropriateness (target ~400 words)
        word_count = len(generated_text.split())
        length_score = 1.0 - abs(word_count - 400) / 400
        scores['length_appropriateness'] = max(length_score, 0.0)
        
        # Paragraph structure (should have 4 paragraphs)
        paragraphs = self._split_into_paragraphs(generated_text)
        structure_score = 1.0 if len(paragraphs) == 4 else 0.8
        scores['structure_appropriateness'] = structure_score
        
        # Citation density (reasonable distribution)
        avg_citation_density = citations_used / max(len(paragraphs), 1)
        density_score = 1.0 if 1.0 <= avg_citation_density <= 3.0 else 0.7
        scores['citation_density'] = density_score
        
        # Overall quality (weighted average)
        scores['overall'] = (
            scores['citation_completeness'] * 0.3 +
            scores['length_appropriateness'] * 0.2 +
            scores['structure_appropriateness'] * 0.2 + 
            scores['citation_density'] * 0.3
        )
        
        return scores
    
    def _get_empirical_validation(self, generated_text: str, research_domain: str) -> Dict[str, Any]:
        """Get empirical validation based on trained patterns."""
        try:
            # Load empirical patterns from trained data
            empirical_patterns = self.empirical_detector.detect_patterns_empirical(
                [], require_statistical_significance=True
            )
            
            paragraphs = self._split_into_paragraphs(generated_text)
            current_paragraph_count = len(paragraphs)
            
            # Find paragraph count pattern
            paragraph_pattern = None
            for pattern in empirical_patterns:
                if pattern.pattern_type == "paragraph_count":
                    paragraph_pattern = pattern
                    break
            
            if paragraph_pattern:
                # Use real empirical data
                mean_paragraphs = paragraph_pattern.statistical_evidence.get('mean', 4.0)
                std_paragraphs = paragraph_pattern.statistical_evidence.get('std', 0.7)
                confidence_interval = paragraph_pattern.confidence_interval
                sample_size = paragraph_pattern.sample_size
                
                # Calculate how well current text matches empirical patterns
                deviation = abs(current_paragraph_count - mean_paragraphs)
                normalized_deviation = deviation / max(std_paragraphs, 1.0)
                pattern_match_score = max(0.0, 1.0 - normalized_deviation / 2.0)
                
                validation = {
                    'paragraph_count': current_paragraph_count,
                    'empirical_optimal_count': f"{mean_paragraphs:.1f}±{std_paragraphs:.1f}",
                    'matches_empirical_structure': deviation <= std_paragraphs,
                    'domain': research_domain,
                    'pattern_confidence': float(pattern_match_score),
                    'empirical_source': f'Trained from {sample_size} published papers',
                    'confidence_interval': confidence_interval,
                    'deviation_score': float(normalized_deviation),
                    'journals_analyzed': paragraph_pattern.journals_analyzed,
                    'validation_score': paragraph_pattern.validation_score
                }
            else:
                # No empirical data available - indicate this clearly
                validation = {
                    'paragraph_count': current_paragraph_count,
                    'empirical_optimal_count': 'No empirical data available',
                    'matches_empirical_structure': False,
                    'domain': research_domain,
                    'pattern_confidence': 0.0,
                    'empirical_source': 'No trained patterns found - train system first',
                    'error': 'Empirical patterns not trained - run training script first'
                }
            
            return validation
            
        except Exception as e:
            logger.warning(f"Could not perform empirical validation: {e}")
            return {
                'error': str(e),
                'empirical_source': 'Validation failed - check empirical pattern training'
            }
    
    def _adjust_text_length(self, text: str, target_words: int) -> str:
        """Adjust text length to approximate target word count."""
        current_words = len(text.split())
        
        if abs(current_words - target_words) <= 50:  # Within acceptable range
            return text
        
        if current_words > target_words * 1.2:  # Too long
            # Simple truncation strategy - would be more sophisticated in production
            words = text.split()
            truncated = ' '.join(words[:target_words])
            # Ensure it ends properly
            if not truncated.endswith('.'):
                truncated += '.'
            return truncated
        elif current_words < target_words * 0.8:  # Too short
            # Simple extension strategy - add transitional phrases
            enhanced_text = text.replace('. ', '. Moreover, ')
            enhanced_text = enhanced_text.replace('However, ', 'However, it is important to note that ')
            return enhanced_text
        
        return text