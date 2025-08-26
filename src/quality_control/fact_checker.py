"""Factual consistency checker for neuroscience content."""

import logging
from typing import List, Dict, Optional
from .models import FactualClaim, ValidationIssue, WarningLevel, ValidationType

logger = logging.getLogger(__name__)


class FactualConsistencyChecker:
    """Checks factual consistency of neuroscience claims."""
    
    def __init__(self):
        """Initialize fact checker with neuroscience knowledge base."""
        self.neuroscience_facts = self._load_neuroscience_facts()
        self.common_errors = self._load_common_errors()
    
    def check_factual_claims(self, text: str, source_papers: List = None) -> List[FactualClaim]:
        """
        Check factual claims in the text for accuracy.
        
        Args:
            text: Text to check for factual claims
            source_papers: Source papers for verification
            
        Returns:
            List of factual claims with verification status
        """
        claims = []
        
        # Extract factual statements
        factual_statements = self._extract_factual_statements(text)
        
        for statement in factual_statements:
            claim = self._verify_factual_claim(statement, source_papers)
            claims.append(claim)
        
        return claims
    
    def _load_neuroscience_facts(self) -> Dict[str, Dict]:
        """Load verified neuroscience facts for validation."""
        return {
            'hippocampus': {
                'functions': ['memory formation', 'spatial navigation', 'learning'],
                'structure': 'part of limbic system',
                'location': 'medial temporal lobe'
            },
            'neurons': {
                'count': 'approximately 86 billion in human brain',
                'types': ['motor neurons', 'sensory neurons', 'interneurons'],
                'structure': ['cell body', 'axon', 'dendrites']
            },
            'brain_weight': {
                'human_adult': '1.4 kg average',
                'percentage_body_weight': '2% of body weight'
            }
        }
    
    def _load_common_errors(self) -> List[Dict]:
        """Load common factual errors in neuroscience writing."""
        return [
            {
                'error': 'we only use 10% of our brain',
                'correction': 'humans use virtually all of their brain',
                'severity': 'critical'
            },
            {
                'error': 'left brain/right brain personality types',
                'correction': 'both hemispheres work together for most functions',
                'severity': 'high'
            }
        ]
    
    def _extract_factual_statements(self, text: str) -> List[str]:
        """Extract statements that make factual claims."""
        # Simplified extraction - look for definitive statements
        sentences = text.split('.')
        factual_statements = []
        
        factual_indicators = [
            'contains', 'consists of', 'is composed of', 'measures',
            'weighs', 'located in', 'responsible for', 'controls'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in factual_indicators):
                factual_statements.append(sentence)
        
        return factual_statements
    
    def _verify_factual_claim(self, statement: str, source_papers: List = None) -> FactualClaim:
        """Verify a single factual claim."""
        statement_lower = statement.lower()
        
        # Check against known facts
        verification_status = 'unknown'
        supporting_evidence = []
        contradicting_evidence = []
        confidence = 0.5
        
        # Check for common errors
        for error in self.common_errors:
            if error['error'].lower() in statement_lower:
                verification_status = 'contradicted'
                contradicting_evidence.append(error['correction'])
                confidence = 0.9
                break
        
        # Check against knowledge base
        for fact_category, facts in self.neuroscience_facts.items():
            if fact_category in statement_lower:
                # Simplified fact checking
                verification_status = 'verified'
                supporting_evidence.append(f"Known fact about {fact_category}")
                confidence = 0.8
                break
        
        return FactualClaim(
            claim_text=statement,
            claim_type=self._classify_claim_type(statement),
            location="factual statement",
            confidence=confidence,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            verification_status=verification_status
        )
    
    def _classify_claim_type(self, claim: str) -> str:
        """Classify the type of factual claim."""
        claim_lower = claim.lower()
        
        if any(term in claim_lower for term in ['hippocampus', 'cortex', 'amygdala', 'brain']):
            return 'anatomical'
        elif any(term in claim_lower for term in ['memory', 'learning', 'cognition']):
            return 'functional'
        elif any(term in claim_lower for term in ['p =', 'r =', 'significant']):
            return 'statistical'
        elif any(term in claim_lower for term in ['fmri', 'eeg', 'pet', 'method']):
            return 'methodological'
        else:
            return 'general'