"""
Draft Generator Module

Provides citation-aware introduction generation capabilities that integrate
specific reference papers with empirically-trained writing patterns.

This module enables users to generate introduction drafts that:
- Use empirical patterns learned from successful papers
- Integrate specific reference papers from their bibliography
- Ensure citation coherence and contextual appropriateness
- Maintain scientific rigor through quality control
"""

from .citation_aware_generator import CitationAwareGenerator
from .reference_integration import ReferenceIntegrator, ReferenceAnalysis
from .draft_validator import DraftValidator, CitationCoherenceValidator

__all__ = [
    'CitationAwareGenerator',
    'ReferenceIntegrator', 
    'ReferenceAnalysis',
    'DraftValidator',
    'CitationCoherenceValidator'
]