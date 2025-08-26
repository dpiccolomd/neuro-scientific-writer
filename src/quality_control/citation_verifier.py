"""Citation verification and context validation."""

import re
import logging
from typing import List, Dict, Optional, Tuple
from .models import CitationValidation, ValidationIssue, WarningLevel, ValidationType

logger = logging.getLogger(__name__)


class CitationVerifier:
    """Verifies citations and their contextual appropriateness."""
    
    def __init__(self):
        """Initialize citation verifier with validation patterns."""
        self.apa_patterns = self._load_apa_patterns()
        self.context_indicators = self._load_context_indicators()
    
    def verify_citation(self, citation: str, context: str, 
                       source_papers: List = None) -> CitationValidation:
        """
        Verify a single citation for accuracy and appropriateness.
        
        Args:
            citation: The citation text to verify
            context: Surrounding text context
            source_papers: Available source papers for cross-reference
            
        Returns:
            CitationValidation with verification results
        """
        issues = []
        
        # Check APA formatting
        formatting_correct = self._check_apa_format(citation)
        if not formatting_correct:
            issues.append(ValidationIssue(
                issue_type=ValidationType.CITATION_VERIFICATION,
                severity=WarningLevel.MODERATE,
                message=f"Citation may not follow APA format: {citation}",
                location="citation format",
                suggestion="Verify APA 7th edition formatting guidelines"
            ))
        
        # Check context appropriateness
        context_appropriate = self._check_context_appropriateness(citation, context)
        if not context_appropriate:
            issues.append(ValidationIssue(
                issue_type=ValidationType.CITATION_VERIFICATION,
                severity=WarningLevel.HIGH,
                message=f"Citation context may be inappropriate: {citation}",
                location="citation context",
                suggestion="Ensure citation directly supports the claim being made"
            ))
        
        # Simplified source checking (would be enhanced in production)
        found_in_sources = source_papers is not None and len(source_papers) > 0
        source_accessible = True  # Would check actual accessibility
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(
            formatting_correct, context_appropriate, found_in_sources
        )
        
        return CitationValidation(
            citation_text=citation,
            found_in_sources=found_in_sources,
            context_appropriate=context_appropriate,
            formatting_correct=formatting_correct,
            source_accessible=source_accessible,
            confidence_score=confidence_score,
            issues=issues
        )
    
    def _load_apa_patterns(self) -> Dict[str, str]:
        """Load APA citation patterns."""
        return {
            'basic': r'\([A-Za-z]+(?:\s+[A-Za-z]+)*,?\s+\d{4}\)',
            'multiple_authors': r'\([A-Za-z]+(?:\s+[A-Za-z]+)*\s+(?:et\s+al\.|&\s+[A-Za-z]+(?:\s+[A-Za-z]+)*),?\s+\d{4}\)',
            'page_numbers': r'\([A-Za-z]+(?:\s+[A-Za-z]+)*,?\s+\d{4},?\s+p\.?\s*\d+\)'
        }
    
    def _load_context_indicators(self) -> Dict[str, List[str]]:
        """Load context indicators for appropriate citation usage."""
        return {
            'supporting': ['showed', 'demonstrated', 'found', 'reported', 'indicated', 'revealed'],
            'contrasting': ['however', 'contrary', 'in contrast', 'conversely', 'nevertheless'],
            'background': ['previous', 'prior', 'earlier', 'established', 'known'],
            'methodology': ['using', 'following', 'according to', 'as described by']
        }
    
    def _check_apa_format(self, citation: str) -> bool:
        """Check if citation follows APA format."""
        for pattern_name, pattern in self.apa_patterns.items():
            if re.match(pattern, citation):
                return True
        return False
    
    def _check_context_appropriateness(self, citation: str, context: str) -> bool:
        """Check if citation is used in appropriate context."""
        context_lower = context.lower()
        
        # Look for appropriate context indicators
        for context_type, indicators in self.context_indicators.items():
            if any(indicator in context_lower for indicator in indicators):
                return True
        
        # Check for claim-making statements
        claim_patterns = [
            r'\b(?:is|are|was|were)\s+',
            r'\b(?:shows?|demonstrates?|indicates?)\s+',
            r'\b(?:suggests?|implies?)\s+'
        ]
        
        return any(re.search(pattern, context_lower) for pattern in claim_patterns)
    
    def _calculate_confidence(self, formatting_correct: bool, 
                            context_appropriate: bool, 
                            found_in_sources: bool) -> float:
        """Calculate confidence score for citation validation."""
        score = 0.0
        
        if formatting_correct:
            score += 0.3
        if context_appropriate:
            score += 0.4
        if found_in_sources:
            score += 0.3
        
        return score