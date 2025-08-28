"""
Draft Validator with Citation Coherence Checking

Provides comprehensive validation of generated introduction drafts,
ensuring citation coherence, logical flow, and scientific accuracy.
"""

import logging
import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .models import GeneratedDraft, CitationCoherenceResult, ValidationReport, ReferenceContext
from quality_control.validator import QualityValidator
from quality_control.citation_verifier import CitationVerifier
from quality_control.models import ValidationResult

logger = logging.getLogger(__name__)


class CitationCoherenceValidator:
    """Validates coherence between citations and text content."""
    
    def __init__(self):
        """Initialize citation coherence validator."""
        self.concept_keywords = self._load_concept_keywords()
        self.assertion_indicators = self._load_assertion_indicators()
        
    def validate_citation_coherence(
        self,
        generated_text: str,
        reference_contexts: List[ReferenceContext],
        paragraph_breakdown: List[str]
    ) -> CitationCoherenceResult:
        """
        Validate coherence between citations and content.
        
        Args:
            generated_text: Generated introduction text
            reference_contexts: Reference contexts used
            paragraph_breakdown: Text split by paragraphs
            
        Returns:
            Citation coherence analysis results
        """
        logger.info("Validating citation coherence")
        
        # Extract citations from text
        citations_in_text = self._extract_citations_with_positions(generated_text)
        
        # Analyze paragraph-level coherence
        paragraph_coherence = self._analyze_paragraph_coherence(
            paragraph_breakdown, reference_contexts, citations_in_text
        )
        
        # Analyze citation-context matches
        citation_context_matches = self._analyze_citation_context_matches(
            generated_text, reference_contexts, citations_in_text
        )
        
        # Analyze logical flow
        logical_flow_score = self._analyze_logical_flow(
            paragraph_breakdown, citations_in_text
        )
        
        # Analyze citation density appropriateness
        citation_density_score = self._analyze_citation_density(
            paragraph_breakdown, citations_in_text
        )
        
        # Calculate overall coherence
        overall_coherence = self._calculate_overall_coherence(
            paragraph_coherence, citation_context_matches, logical_flow_score, citation_density_score
        )
        
        # Identify issues and recommendations
        issues, recommendations = self._identify_issues_and_recommendations(
            paragraph_coherence, citation_context_matches, logical_flow_score, citation_density_score
        )
        
        result = CitationCoherenceResult(
            overall_coherence=overall_coherence,
            paragraph_coherence=paragraph_coherence,
            citation_context_matches=citation_context_matches,
            logical_flow_score=logical_flow_score,
            citation_density_appropriateness=citation_density_score,
            issues_identified=issues,
            recommendations=recommendations
        )
        
        logger.info(f"Citation coherence validation complete: "
                   f"overall={overall_coherence:.3f}, "
                   f"flow={logical_flow_score:.3f}, "
                   f"density={citation_density_score:.3f}")
        
        return result
    
    def _extract_citations_with_positions(self, text: str) -> List[Tuple[str, int, int]]:
        """Extract citations with their positions in text."""
        citations = []
        
        # Pattern to match citations in parentheses
        citation_pattern = r'\(([^)]+\d{4}[^)]*)\)'
        
        for match in re.finditer(citation_pattern, text):
            citation_text = match.group(1)
            start_pos = match.start()
            end_pos = match.end()
            
            # Split multiple citations within same parentheses
            individual_citations = [c.strip() for c in citation_text.split(';')]
            
            for citation in individual_citations:
                if self._is_valid_citation_format(citation):
                    citations.append((citation, start_pos, end_pos))
        
        return citations
    
    def _is_valid_citation_format(self, citation: str) -> bool:
        """Check if citation follows valid format (Author, Year)."""
        # Basic check for author name and year
        has_year = re.search(r'\d{4}', citation)
        has_author = len(citation.split()) >= 2 and any(c.isalpha() for c in citation)
        
        return has_year and has_author
    
    def _analyze_paragraph_coherence(
        self,
        paragraphs: List[str],
        reference_contexts: List[ReferenceContext],
        citations_in_text: List[Tuple[str, int, int]]
    ) -> Dict[int, float]:
        """Analyze coherence within each paragraph."""
        paragraph_coherence = {}
        text_so_far = 0
        
        for i, paragraph in enumerate(paragraphs, 1):
            para_start = text_so_far
            para_end = text_so_far + len(paragraph)
            
            # Find citations in this paragraph
            para_citations = [
                (citation, pos) for citation, start_pos, end_pos in citations_in_text
                if para_start <= start_pos <= para_end
            ]
            
            # Calculate coherence score for this paragraph
            coherence_score = self._calculate_paragraph_coherence_score(
                paragraph, para_citations, reference_contexts
            )
            
            paragraph_coherence[i] = coherence_score
            text_so_far = para_end + 2  # Account for paragraph breaks
        
        return paragraph_coherence
    
    def _calculate_paragraph_coherence_score(
        self,
        paragraph: str,
        paragraph_citations: List[Tuple[str, int]],
        reference_contexts: List[ReferenceContext]
    ) -> float:
        """Calculate coherence score for a single paragraph."""
        if not paragraph_citations:
            return 0.8  # Neutral score for paragraphs without citations
        
        scores = []
        
        for citation, _ in paragraph_citations:
            # Find corresponding reference context
            ref_context = self._find_reference_context_by_citation(citation, reference_contexts)
            if not ref_context:
                scores.append(0.5)  # Neutral for unknown references
                continue
            
            # Check concept alignment
            concept_score = self._calculate_concept_alignment(paragraph, ref_context)
            
            # Check assertion appropriateness
            assertion_score = self._calculate_assertion_appropriateness(paragraph, ref_context)
            
            # Combine scores
            citation_score = (concept_score + assertion_score) / 2
            scores.append(citation_score)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _calculate_concept_alignment(self, paragraph: str, ref_context: ReferenceContext) -> float:
        """Calculate how well paragraph concepts align with reference concepts."""
        if not ref_context.key_concepts:
            return 0.5
        
        paragraph_lower = paragraph.lower()
        concept_matches = 0
        
        for concept in ref_context.key_concepts:
            if concept.lower() in paragraph_lower:
                concept_matches += 1
        
        alignment_score = min(concept_matches / len(ref_context.key_concepts), 1.0)
        return alignment_score
    
    def _calculate_assertion_appropriateness(self, paragraph: str, ref_context: ReferenceContext) -> float:
        """Calculate appropriateness of assertions for this reference type."""
        paragraph_lower = paragraph.lower()
        
        # Check for appropriate assertion indicators based on reference role
        role_indicators = {
            'foundational': ['established', 'fundamental', 'foundation', 'basis'],
            'supporting': ['demonstrated', 'showed', 'found', 'evidence'],
            'contrasting': ['however', 'alternatively', 'contrast', 'different'],
            'gap_identifying': ['limited', 'unclear', 'unknown', 'gap'],
            'methodological': ['method', 'approach', 'technique', 'analysis']
        }
        
        expected_indicators = role_indicators.get(ref_context.role.value, [])
        
        matches = sum(1 for indicator in expected_indicators if indicator in paragraph_lower)
        
        if expected_indicators:
            return min(matches / len(expected_indicators), 1.0)
        else:
            return 0.7  # Default score
    
    def _analyze_citation_context_matches(
        self,
        text: str,
        reference_contexts: List[ReferenceContext],
        citations_in_text: List[Tuple[str, int, int]]
    ) -> Dict[str, float]:
        """Analyze how well each citation matches its surrounding context."""
        context_matches = {}
        
        for citation, start_pos, end_pos in citations_in_text:
            # Get surrounding context (50 chars before and after)
            context_start = max(0, start_pos - 50)
            context_end = min(len(text), end_pos + 50)
            surrounding_context = text[context_start:context_end]
            
            # Find reference context
            ref_context = self._find_reference_context_by_citation(citation, reference_contexts)
            if not ref_context:
                context_matches[citation] = 0.5
                continue
            
            # Calculate match score
            match_score = self._calculate_context_match_score(surrounding_context, ref_context)
            context_matches[citation] = match_score
        
        return context_matches
    
    def _calculate_context_match_score(self, surrounding_context: str, ref_context: ReferenceContext) -> float:
        """Calculate how well surrounding context matches reference content."""
        context_lower = surrounding_context.lower()
        
        # Check concept matches
        concept_matches = sum(
            1 for concept in ref_context.key_concepts 
            if concept.lower() in context_lower
        )
        
        # Check finding matches
        finding_matches = sum(
            1 for finding in ref_context.key_findings
            if any(word in context_lower for word in finding.lower().split()[:5])
        )
        
        # Calculate score based on matches
        total_possible_matches = len(ref_context.key_concepts) + len(ref_context.key_findings)
        if total_possible_matches == 0:
            return 0.6
        
        actual_matches = concept_matches + finding_matches
        return min(actual_matches / total_possible_matches, 1.0)
    
    def _analyze_logical_flow(
        self,
        paragraphs: List[str],
        citations_in_text: List[Tuple[str, int, int]]
    ) -> float:
        """Analyze logical flow between paragraphs."""
        if len(paragraphs) < 2:
            return 1.0
        
        flow_scores = []
        
        for i in range(len(paragraphs) - 1):
            current_para = paragraphs[i]
            next_para = paragraphs[i + 1]
            
            # Check for transition indicators
            transition_score = self._calculate_transition_score(current_para, next_para)
            flow_scores.append(transition_score)
        
        return sum(flow_scores) / len(flow_scores) if flow_scores else 0.7
    
    def _calculate_transition_score(self, current_para: str, next_para: str) -> float:
        """Calculate transition quality between two paragraphs."""
        # Look for transition words/phrases
        transition_indicators = [
            'however', 'moreover', 'furthermore', 'additionally', 'nevertheless',
            'therefore', 'consequently', 'meanwhile', 'similarly', 'in contrast',
            'despite', 'although', 'while', 'whereas', 'given that'
        ]
        
        next_para_lower = next_para.lower()
        
        # Check if next paragraph starts with transition
        starts_with_transition = any(
            next_para_lower.strip().startswith(indicator)
            for indicator in transition_indicators
        )
        
        # Check for conceptual continuity (shared concepts)
        current_words = set(current_para.lower().split())
        next_words = set(next_para.lower().split())
        shared_concepts = len(current_words & next_words & set(self.concept_keywords))
        
        # Calculate score
        transition_score = 0.5  # Base score
        if starts_with_transition:
            transition_score += 0.3
        if shared_concepts > 2:
            transition_score += 0.2
        
        return min(transition_score, 1.0)
    
    def _analyze_citation_density(
        self,
        paragraphs: List[str],
        citations_in_text: List[Tuple[str, int, int]]
    ) -> float:
        """Analyze appropriateness of citation density."""
        if not paragraphs:
            return 0.5
        
        # Calculate citations per paragraph
        text_so_far = 0
        paragraph_citation_counts = []
        
        for paragraph in paragraphs:
            para_start = text_so_far
            para_end = text_so_far + len(paragraph)
            
            para_citations = sum(
                1 for _, start_pos, _ in citations_in_text
                if para_start <= start_pos <= para_end
            )
            
            paragraph_citation_counts.append(para_citations)
            text_so_far = para_end + 2
        
        # Evaluate density appropriateness
        density_scores = []
        
        for i, citation_count in enumerate(paragraph_citation_counts):
            paragraph = paragraphs[i]
            sentences = len([s for s in paragraph.split('.') if s.strip()])
            
            if sentences == 0:
                density_scores.append(0.5)
                continue
            
            citations_per_sentence = citation_count / sentences
            
            # Optimal range: 0.2-0.8 citations per sentence
            if 0.2 <= citations_per_sentence <= 0.8:
                density_score = 1.0
            elif citations_per_sentence < 0.2:
                density_score = max(0.3, citations_per_sentence / 0.2 * 0.7)
            else:  # Too dense
                density_score = max(0.3, 1.0 - (citations_per_sentence - 0.8) * 0.5)
            
            density_scores.append(density_score)
        
        return sum(density_scores) / len(density_scores) if density_scores else 0.5
    
    def _calculate_overall_coherence(
        self,
        paragraph_coherence: Dict[int, float],
        citation_context_matches: Dict[str, float],
        logical_flow_score: float,
        citation_density_score: float
    ) -> float:
        """Calculate overall coherence score."""
        # Average paragraph coherence
        avg_paragraph_coherence = sum(paragraph_coherence.values()) / len(paragraph_coherence) if paragraph_coherence else 0.5
        
        # Average citation context matches
        avg_context_matches = sum(citation_context_matches.values()) / len(citation_context_matches) if citation_context_matches else 0.5
        
        # Weighted average
        overall = (
            avg_paragraph_coherence * 0.3 +
            avg_context_matches * 0.3 +
            logical_flow_score * 0.2 +
            citation_density_score * 0.2
        )
        
        return overall
    
    def _identify_issues_and_recommendations(
        self,
        paragraph_coherence: Dict[int, float],
        citation_context_matches: Dict[str, float],
        logical_flow_score: float,
        citation_density_score: float
    ) -> Tuple[List[str], List[str]]:
        """Identify issues and generate recommendations."""
        issues = []
        recommendations = []
        
        # Check paragraph coherence
        low_coherence_paras = [
            para_num for para_num, score in paragraph_coherence.items()
            if score < 0.6
        ]
        if low_coherence_paras:
            issues.append(f"Low citation coherence in paragraph(s): {', '.join(map(str, low_coherence_paras))}")
            recommendations.append("Review citations in low-coherence paragraphs to ensure they support the claims being made")
        
        # Check citation context matches
        low_match_citations = [
            citation for citation, score in citation_context_matches.items()
            if score < 0.5
        ]
        if low_match_citations:
            issues.append(f"Poor context matches for citations: {', '.join(low_match_citations[:3])}")
            recommendations.append("Ensure citations are placed near text that discusses related concepts or findings")
        
        # Check logical flow
        if logical_flow_score < 0.6:
            issues.append("Poor logical flow between paragraphs")
            recommendations.append("Add transition phrases and ensure conceptual continuity between paragraphs")
        
        # Check citation density
        if citation_density_score < 0.5:
            issues.append("Inappropriate citation density")
            if citation_density_score < 0.3:
                recommendations.append("Add more citations to support claims, especially in literature review sections")
            else:
                recommendations.append("Redistribute citations more evenly or reduce citation density in over-cited sections")
        
        return issues, recommendations
    
    def _find_reference_context_by_citation(
        self, 
        citation: str, 
        reference_contexts: List[ReferenceContext]
    ) -> Optional[ReferenceContext]:
        """Find reference context matching a citation."""
        # Simple matching by looking for author name in citation
        citation_words = citation.lower().split()
        
        for ref_context in reference_contexts:
            ref_citation = ref_context.citation_format.lower()
            
            # Check if citation words appear in reference citation
            if any(word in ref_citation for word in citation_words if len(word) > 3):
                return ref_context
        
        return None
    
    def _load_concept_keywords(self) -> List[str]:
        """Load neuroscience concept keywords."""
        return [
            'neural', 'brain', 'cognitive', 'memory', 'attention', 'neurotransmitter',
            'synaptic', 'cortical', 'hippocampus', 'amygdala', 'plasticity', 'network',
            'activation', 'connectivity', 'processing', 'function', 'mechanism',
            'pathway', 'circuit', 'oscillation', 'synchronization', 'modulation'
        ]
    
    def _load_assertion_indicators(self) -> List[str]:
        """Load indicators of different types of assertions."""
        return [
            'demonstrated', 'showed', 'found', 'revealed', 'indicated',
            'suggested', 'proposed', 'hypothesized', 'established',
            'confirmed', 'supported', 'evidence', 'results'
        ]


class DraftValidator:
    """Comprehensive validator for generated introduction drafts."""
    
    def __init__(self):
        """Initialize draft validator."""
        self.quality_validator = QualityValidator()
        self.citation_verifier = CitationVerifier()
        self.citation_coherence_validator = CitationCoherenceValidator()
        
    def validate_draft(self, draft: GeneratedDraft) -> ValidationReport:
        """
        Perform comprehensive validation of generated draft.
        
        Args:
            draft: Generated introduction draft
            
        Returns:
            Complete validation report
        """
        logger.info(f"Validating draft: {draft.draft_id}")
        
        # Validate citation coherence
        citation_coherence = self.citation_coherence_validator.validate_citation_coherence(
            draft.generated_text,
            draft.integration_plan.reference_contexts,
            draft.paragraph_breakdown
        )
        
        # Use existing quality validation
        quality_result = self.quality_validator.validate_text(draft.generated_text)
        
        # Extract specific scores
        empirical_pattern_compliance = draft.quality_scores.get('overall', 0.0)
        factual_accuracy = quality_result.factual_consistency if hasattr(quality_result, 'factual_consistency') else 0.8
        statistical_validity = quality_result.statistical_validity if hasattr(quality_result, 'statistical_validity') else 0.8
        terminology_appropriateness = quality_result.terminology_score if hasattr(quality_result, 'terminology_score') else 0.8
        
        # Calculate overall quality
        overall_quality = (
            empirical_pattern_compliance * 0.25 +
            citation_coherence.overall_coherence * 0.25 +
            factual_accuracy * 0.2 +
            statistical_validity * 0.15 +
            terminology_appropriateness * 0.15
        )
        
        # Identify critical and minor issues
        critical_issues = []
        minor_issues = []
        recommendations = []
        
        # Add citation coherence issues
        if citation_coherence.overall_coherence < 0.6:
            critical_issues.extend(citation_coherence.issues_identified)
        elif citation_coherence.overall_coherence < 0.8:
            minor_issues.extend(citation_coherence.issues_identified)
        
        recommendations.extend(citation_coherence.recommendations)
        
        # Add quality validation issues
        if hasattr(quality_result, 'issues'):
            for issue in quality_result.issues:
                if issue.severity == 'high':
                    critical_issues.append(issue.message)
                else:
                    minor_issues.append(issue.message)
        
        # Determine if ready for submission
        ready_for_submission = (
            overall_quality >= 0.8 and
            len(critical_issues) == 0 and
            citation_coherence.is_publication_ready
        )
        
        report = ValidationReport(
            draft_id=draft.draft_id,
            empirical_pattern_compliance=empirical_pattern_compliance,
            citation_coherence=citation_coherence,
            factual_accuracy=factual_accuracy,
            statistical_validity=statistical_validity,
            terminology_appropriateness=terminology_appropriateness,
            overall_quality=overall_quality,
            critical_issues=critical_issues,
            minor_issues=minor_issues,
            recommendations=recommendations,
            ready_for_submission=ready_for_submission
        )
        
        logger.info(f"Draft validation complete: overall_quality={overall_quality:.3f}, "
                   f"ready_for_submission={ready_for_submission}")
        
        return report