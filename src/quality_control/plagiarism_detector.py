"""Plagiarism detection for scientific manuscripts."""

import re
import logging
from typing import List, Dict, Optional, Set
from .models import PlagiarismResult, WarningLevel

logger = logging.getLogger(__name__)


class PlagiarismDetector:
    """Detects potential plagiarism in scientific writing."""
    
    def __init__(self):
        """Initialize plagiarism detector."""
        self.common_phrases = self._load_common_phrases()
        self.suspicious_patterns = self._load_suspicious_patterns()
        self.similarity_threshold = 0.7
    
    def detect_plagiarism(self, text: str, source_texts: List[str] = None) -> List[PlagiarismResult]:
        """
        Detect potential plagiarism in the text.
        
        Args:
            text: Text to check for plagiarism
            source_texts: Known source texts to compare against
            
        Returns:
            List of potential plagiarism results
        """
        results = []
        
        # Split text into sentences for analysis
        sentences = self._split_into_sentences(text)
        
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 20:  # Skip very short sentences
                continue
            
            # Check against common academic phrases
            common_phrase_result = self._check_common_phrases(sentence, i)
            if common_phrase_result:
                results.append(common_phrase_result)
            
            # Check for suspicious patterns
            suspicious_result = self._check_suspicious_patterns(sentence, i)
            if suspicious_result:
                results.append(suspicious_result)
            
            # Check against source texts if provided
            if source_texts:
                source_similarity = self._check_source_similarity(sentence, source_texts, i)
                if source_similarity:
                    results.append(source_similarity)
        
        return results
    
    def _load_common_phrases(self) -> Dict[str, float]:
        """Load common academic phrases and their risk levels."""
        return {
            'it has been well established that': 0.8,
            'extensive research has demonstrated': 0.7,
            'it is widely accepted that': 0.6,
            'numerous studies have shown': 0.5,
            'previous research has indicated': 0.4,
            'it is generally recognized': 0.6,
            'the literature suggests that': 0.3,
            'according to the literature': 0.3,
            'research has consistently shown': 0.5,
            'it is important to note that': 0.4
        }
    
    def _load_suspicious_patterns(self) -> List[Dict[str, any]]:
        """Load patterns that might indicate plagiarism."""
        return [
            {
                'pattern': r'\b(?:it|this)\s+(?:has been|is)\s+(?:well|widely)\s+(?:established|known|accepted)\b',
                'description': 'Overly formal/textbook language',
                'risk_level': 0.6
            },
            {
                'pattern': r'\b(?:numerous|extensive|considerable)\s+(?:studies|research|investigations)\b',
                'description': 'Generic reference to studies',
                'risk_level': 0.5
            },
            {
                'pattern': r'\bmoreover,?\s+it\s+(?:is|has been|should be)\s+noted\s+that\b',
                'description': 'Formal transitional phrases',
                'risk_level': 0.4
            },
            {
                'pattern': r'\bin\s+(?:conclusion|summary),?\s+(?:it\s+(?:can\s+be|is)|the\s+present)\b',
                'description': 'Formulaic conclusion phrases',
                'risk_level': 0.3
            }
        ]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - could be enhanced
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _check_common_phrases(self, sentence: str, position: int) -> Optional[PlagiarismResult]:
        """Check sentence against common academic phrases."""
        sentence_lower = sentence.lower()
        
        for phrase, risk_level in self.common_phrases.items():
            if phrase in sentence_lower:
                severity = self._risk_level_to_severity(risk_level)
                
                return PlagiarismResult(
                    text_segment=sentence,
                    similarity_score=risk_level,
                    potential_sources=['common academic phrases'],
                    is_problematic=risk_level > 0.6,
                    severity=severity
                )
        
        return None
    
    def _check_suspicious_patterns(self, sentence: str, position: int) -> Optional[PlagiarismResult]:
        """Check for suspicious writing patterns."""
        sentence_lower = sentence.lower()
        
        for pattern_info in self.suspicious_patterns:
            if re.search(pattern_info['pattern'], sentence_lower):
                risk_level = pattern_info['risk_level']
                severity = self._risk_level_to_severity(risk_level)
                
                return PlagiarismResult(
                    text_segment=sentence,
                    similarity_score=risk_level,
                    potential_sources=[pattern_info['description']],
                    is_problematic=risk_level > 0.5,
                    severity=severity
                )
        
        return None
    
    def _check_source_similarity(self, sentence: str, source_texts: List[str], 
                                position: int) -> Optional[PlagiarismResult]:
        """Check similarity against provided source texts."""
        sentence_lower = sentence.lower()
        max_similarity = 0.0
        similar_sources = []
        
        for i, source_text in enumerate(source_texts):
            source_lower = source_text.lower()
            
            # Simple similarity check - look for overlapping phrases
            similarity = self._calculate_text_similarity(sentence_lower, source_lower)
            
            if similarity > self.similarity_threshold:
                max_similarity = max(max_similarity, similarity)
                similar_sources.append(f'source_document_{i+1}')
        
        if max_similarity > self.similarity_threshold:
            severity = self._risk_level_to_severity(max_similarity)
            
            return PlagiarismResult(
                text_segment=sentence,
                similarity_score=max_similarity,
                potential_sources=similar_sources,
                is_problematic=True,
                severity=severity
            )
        
        return None
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity between two texts."""
        # Very simplified similarity calculation
        # In production, would use more sophisticated methods like cosine similarity
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        # Jaccard similarity
        return len(intersection) / len(union) if union else 0.0
    
    def _risk_level_to_severity(self, risk_level: float) -> WarningLevel:
        """Convert risk level to warning severity."""
        if risk_level >= 0.8:
            return WarningLevel.CRITICAL
        elif risk_level >= 0.6:
            return WarningLevel.HIGH
        elif risk_level >= 0.4:
            return WarningLevel.MODERATE
        else:
            return WarningLevel.LOW
    
    def check_self_plagiarism(self, text: str, previous_works: List[str]) -> List[PlagiarismResult]:
        """Check for self-plagiarism against previous works."""
        # Similar to source similarity but with different thresholds
        return self._check_source_similarity(text, previous_works, 0)