"""Advanced text analysis for neuroscience documents using NLP."""

import re
import time
import logging
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict
import string

# Note: These would normally use spacy and other NLP libraries
# For now, implementing rule-based analysis that can be enhanced later

from .models import (
    AnalysisResult, NeuroTerm, SentenceStructure, WritingPattern,
    StyleCharacteristics, TerminologyAnalysis, CoherenceMetrics,
    SentenceType, WritingStyle
)
from .exceptions import AnalysisError, TerminologyExtractionError, SentenceAnalysisError

logger = logging.getLogger(__name__)


class NeuroTextAnalyzer:
    """Advanced text analyzer specialized for neuroscience literature."""
    
    def __init__(self):
        """Initialize the analyzer with neuroscience-specific knowledge."""
        self.neuro_terminology = self._load_neuroscience_terminology()
        self.sentence_patterns = self._load_sentence_patterns()
        self.transition_words = self._load_transition_words()
        self.technical_indicators = self._load_technical_indicators()
        
    def analyze_text(self, text: str, text_id: str = None, 
                    source_type: str = "document") -> AnalysisResult:
        """
        Perform comprehensive analysis of neuroscience text.
        
        Args:
            text: Text to analyze
            text_id: Identifier for the text
            source_type: Type of source ('document', 'section', 'paragraph')
            
        Returns:
            Complete analysis result
            
        Raises:
            AnalysisError: If analysis fails
        """
        start_time = time.time()
        
        if not text.strip():
            raise AnalysisError("Text cannot be empty")
        
        text_id = text_id or f"{source_type}_{hash(text)}"
        
        try:
            logger.info(f"Starting analysis of {source_type}: {text_id}")
            
            # Split into sentences
            sentences_text = self._split_sentences(text)
            
            # Analyze each sentence
            sentences = []
            for i, sent_text in enumerate(sentences_text):
                try:
                    sentence_analysis = self._analyze_sentence(sent_text, i)
                    sentences.append(sentence_analysis)
                except Exception as e:
                    logger.warning(f"Failed to analyze sentence {i}: {e}")
                    # Create basic sentence structure for failed analysis
                    sentences.append(SentenceStructure(
                        text=sent_text,
                        sentence_type=SentenceType.UNKNOWN,
                        word_count=len(sent_text.split()),
                        complexity_score=0.5,
                        passive_voice=False,
                        citation_count=0,
                        technical_term_count=0,
                        readability_score=0.5,
                        grammatical_features={}
                    ))
            
            # Extract neuroscience terminology
            neuro_terms = self._extract_terminology(text)
            
            # Detect writing patterns
            writing_patterns = self._detect_writing_patterns(text, sentences)
            
            # Analyze style characteristics
            style_characteristics = self._analyze_style(sentences, neuro_terms)
            
            # Analyze terminology usage
            terminology_analysis = self._analyze_terminology_usage(neuro_terms, len(text.split()))
            
            # Calculate coherence metrics
            coherence_metrics = self._calculate_coherence(text, sentences)
            
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(
                style_characteristics, terminology_analysis, coherence_metrics
            )
            
            processing_time = time.time() - start_time
            
            result = AnalysisResult(
                text_id=text_id,
                source_type=source_type,
                sentences=sentences,
                neuro_terms=neuro_terms,
                writing_patterns=writing_patterns,
                style_characteristics=style_characteristics,
                terminology_analysis=terminology_analysis,
                coherence_metrics=coherence_metrics,
                overall_quality_score=quality_score,
                processing_time=processing_time
            )
            
            logger.info(f"Analysis completed for {text_id} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed for {text_id}: {e}")
            raise AnalysisError(f"Failed to analyze text: {e}", text[:200])
    
    def _load_neuroscience_terminology(self) -> Dict[str, Dict[str, any]]:
        """Load comprehensive neuroscience terminology with categories."""
        return {
            # Neuroanatomy
            'hippocampus': {'category': 'anatomy', 'complexity': 'basic'},
            'amygdala': {'category': 'anatomy', 'complexity': 'basic'},
            'prefrontal cortex': {'category': 'anatomy', 'complexity': 'intermediate'},
            'basal ganglia': {'category': 'anatomy', 'complexity': 'intermediate'},
            'cerebellum': {'category': 'anatomy', 'complexity': 'basic'},
            'brainstem': {'category': 'anatomy', 'complexity': 'basic'},
            'thalamus': {'category': 'anatomy', 'complexity': 'basic'},
            'hypothalamus': {'category': 'anatomy', 'complexity': 'intermediate'},
            'pituitary gland': {'category': 'anatomy', 'complexity': 'basic'},
            'corpus callosum': {'category': 'anatomy', 'complexity': 'intermediate'},
            'ventricles': {'category': 'anatomy', 'complexity': 'intermediate'},
            'meninges': {'category': 'anatomy', 'complexity': 'basic'},
            
            # Cellular components
            'neuron': {'category': 'cellular', 'complexity': 'basic'},
            'axon': {'category': 'cellular', 'complexity': 'basic'},
            'dendrite': {'category': 'cellular', 'complexity': 'basic'},
            'synapse': {'category': 'cellular', 'complexity': 'basic'},
            'myelin': {'category': 'cellular', 'complexity': 'intermediate'},
            'oligodendrocyte': {'category': 'cellular', 'complexity': 'advanced'},
            'astrocyte': {'category': 'cellular', 'complexity': 'intermediate'},
            'microglia': {'category': 'cellular', 'complexity': 'intermediate'},
            'soma': {'category': 'cellular', 'complexity': 'intermediate'},
            'axon hillock': {'category': 'cellular', 'complexity': 'advanced'},
            
            # Neurotransmitters
            'dopamine': {'category': 'neurotransmitter', 'complexity': 'basic'},
            'serotonin': {'category': 'neurotransmitter', 'complexity': 'basic'},
            'norepinephrine': {'category': 'neurotransmitter', 'complexity': 'intermediate'},
            'acetylcholine': {'category': 'neurotransmitter', 'complexity': 'intermediate'},
            'glutamate': {'category': 'neurotransmitter', 'complexity': 'intermediate'},
            'gaba': {'category': 'neurotransmitter', 'complexity': 'intermediate'},
            'glycine': {'category': 'neurotransmitter', 'complexity': 'advanced'},
            'histamine': {'category': 'neurotransmitter', 'complexity': 'advanced'},
            
            # Functions and processes
            'neuroplasticity': {'category': 'function', 'complexity': 'intermediate'},
            'neurogenesis': {'category': 'function', 'complexity': 'advanced'},
            'synaptic transmission': {'category': 'function', 'complexity': 'intermediate'},
            'action potential': {'category': 'function', 'complexity': 'intermediate'},
            'membrane potential': {'category': 'function', 'complexity': 'advanced'},
            'long-term potentiation': {'category': 'function', 'complexity': 'advanced'},
            'long-term depression': {'category': 'function', 'complexity': 'advanced'},
            'memory consolidation': {'category': 'function', 'complexity': 'intermediate'},
            'executive function': {'category': 'function', 'complexity': 'intermediate'},
            'working memory': {'category': 'function', 'complexity': 'basic'},
            
            # Techniques and methods
            'fmri': {'category': 'technique', 'complexity': 'basic'},
            'pet scan': {'category': 'technique', 'complexity': 'basic'},
            'eeg': {'category': 'technique', 'complexity': 'basic'},
            'meg': {'category': 'technique', 'complexity': 'intermediate'},
            'dti': {'category': 'technique', 'complexity': 'advanced'},
            'electrophysiology': {'category': 'technique', 'complexity': 'advanced'},
            'optogenetics': {'category': 'technique', 'complexity': 'advanced'},
            'calcium imaging': {'category': 'technique', 'complexity': 'advanced'},
            'patch clamp': {'category': 'technique', 'complexity': 'advanced'},
            
            # Pathology
            'alzheimer': {'category': 'pathology', 'complexity': 'basic'},
            'parkinson': {'category': 'pathology', 'complexity': 'basic'},
            'epilepsy': {'category': 'pathology', 'complexity': 'basic'},
            'stroke': {'category': 'pathology', 'complexity': 'basic'},
            'traumatic brain injury': {'category': 'pathology', 'complexity': 'intermediate'},
            'multiple sclerosis': {'category': 'pathology', 'complexity': 'intermediate'},
            'huntington': {'category': 'pathology', 'complexity': 'intermediate'},
            'glioblastoma': {'category': 'pathology', 'complexity': 'advanced'},
            'meningioma': {'category': 'pathology', 'complexity': 'intermediate'},
            'hydrocephalus': {'category': 'pathology', 'complexity': 'intermediate'}
        }
    
    def _load_sentence_patterns(self) -> Dict[SentenceType, List[str]]:
        """Load patterns for identifying sentence types."""
        return {
            SentenceType.HYPOTHESIS: [
                r'we hypothes[ie]s?[ez]?d?\s+that',
                r'our hypothesis\s+(?:is|was)',
                r'we predict(?:ed)?\s+that',
                r'it is hypothesized',
                r'we propose\s+that'
            ],
            SentenceType.OBJECTIVE: [
                r'the (?:aim|goal|objective|purpose)\s+(?:of|was)',
                r'we aimed\s+to',
                r'our objective\s+was',
                r'the present study\s+(?:aimed|investigated)',
                r'this study\s+(?:examined|investigated|explored)'
            ],
            SentenceType.METHOD_DESCRIPTION: [
                r'we (?:used|employed|utilized|performed|conducted)',
                r'(?:participants|subjects|patients)\s+(?:were|underwent)',
                r'data (?:were|was)\s+(?:collected|analyzed)',
                r'statistical analysis',
                r'(?:fmri|eeg|pet)\s+(?:data|scans?)\s+(?:were|was)'
            ],
            SentenceType.RESULT_STATEMENT: [
                r'we found\s+that',
                r'results?\s+(?:showed?|demonstrated|indicated|revealed)',
                r'analysis\s+revealed',
                r'(?:significant|no)\s+(?:difference|correlation|effect)',
                r'there (?:was|were)\s+(?:a\s+)?(?:significant|no)'
            ],
            SentenceType.CONCLUSION: [
                r'we conclude\s+that',
                r'in conclusion',
                r'these (?:findings|results)\s+suggest',
                r'our (?:findings|results)\s+(?:demonstrate|indicate|suggest)',
                r'taken together'
            ],
            SentenceType.CITATION_CONTEXT: [
                r'(?:previous|prior)\s+(?:studies|research|work)',
                r'as (?:reported|shown|demonstrated)\s+(?:by|in)',
                r'consistent with\s+(?:previous|prior)',
                r'in agreement with',
                r'contrary to\s+(?:previous|prior)'
            ]
        }
    
    def _load_transition_words(self) -> Set[str]:
        """Load transition words and phrases."""
        return {
            'however', 'nevertheless', 'furthermore', 'moreover', 'therefore',
            'consequently', 'additionally', 'similarly', 'conversely', 'nonetheless',
            'meanwhile', 'subsequently', 'previously', 'initially', 'finally',
            'in contrast', 'in addition', 'as a result', 'on the other hand',
            'taken together', 'in summary', 'in conclusion'
        }
    
    def _load_technical_indicators(self) -> Set[str]:
        """Load indicators of technical/formal language."""
        return {
            'demonstrated', 'investigated', 'analyzed', 'examined', 'assessed',
            'evaluated', 'determined', 'established', 'observed', 'identified',
            'characterized', 'quantified', 'measured', 'calculated', 'estimated',
            'significant', 'correlation', 'regression', 'statistical', 'hypothesis',
            'methodology', 'protocol', 'procedure', 'intervention', 'treatment'
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using rule-based approach."""
        # Basic sentence splitting - could be enhanced with spaCy
        sentences = re.split(r'[.!?]+\s+', text)
        # Clean up
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _analyze_sentence(self, sentence: str, position: int) -> SentenceStructure:
        """Analyze individual sentence structure and characteristics."""
        try:
            # Basic analysis
            words = sentence.split()
            word_count = len(words)
            
            # Detect sentence type
            sentence_type = self._classify_sentence_type(sentence)
            
            # Calculate complexity (simplified)
            complexity_score = self._calculate_sentence_complexity(sentence, words)
            
            # Check for passive voice (simplified)
            passive_voice = self._detect_passive_voice(sentence)
            
            # Count citations
            citation_count = len(re.findall(r'\([^)]*\d{4}[^)]*\)', sentence))
            
            # Count technical terms
            technical_term_count = self._count_technical_terms(sentence)
            
            # Calculate readability (simplified Flesch score approximation)
            readability_score = self._calculate_readability(sentence, words)
            
            # Extract grammatical features
            grammatical_features = self._extract_grammatical_features(sentence, words)
            
            return SentenceStructure(
                text=sentence,
                sentence_type=sentence_type,
                word_count=word_count,
                complexity_score=complexity_score,
                passive_voice=passive_voice,
                citation_count=citation_count,
                technical_term_count=technical_term_count,
                readability_score=readability_score,
                grammatical_features=grammatical_features
            )
            
        except Exception as e:
            raise SentenceAnalysisError(f"Failed to analyze sentence: {e}", sentence)
    
    def _classify_sentence_type(self, sentence: str) -> SentenceType:
        """Classify sentence type based on patterns."""
        sentence_lower = sentence.lower()
        
        for sent_type, patterns in self.sentence_patterns.items():
            for pattern in patterns:
                if re.search(pattern, sentence_lower):
                    return sent_type
        
        return SentenceType.UNKNOWN
    
    def _calculate_sentence_complexity(self, sentence: str, words: List[str]) -> float:
        """Calculate sentence complexity score (0-1)."""
        # Factors contributing to complexity
        word_count = len(words)
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        subordinate_clauses = len(re.findall(r'\b(?:that|which|who|where|when|while|although|because|if|since)\b', sentence.lower()))
        punctuation_count = len(re.findall(r'[,:;()]', sentence))
        
        # Normalize and combine factors
        word_complexity = min(word_count / 30.0, 1.0)  # Max at 30 words
        length_complexity = min(avg_word_length / 8.0, 1.0)  # Max at 8 chars
        clause_complexity = min(subordinate_clauses / 3.0, 1.0)  # Max at 3 clauses
        punct_complexity = min(punctuation_count / 5.0, 1.0)  # Max at 5 punctuation marks
        
        complexity = (word_complexity * 0.3 + length_complexity * 0.2 + 
                     clause_complexity * 0.3 + punct_complexity * 0.2)
        
        return min(complexity, 1.0)
    
    def _detect_passive_voice(self, sentence: str) -> bool:
        """Detect passive voice construction."""
        # Simplified passive voice detection
        passive_patterns = [
            r'\b(?:was|were|is|are|been|being)\s+\w*ed\b',
            r'\b(?:was|were|is|are|been|being)\s+(?:observed|found|shown|demonstrated|reported)\b'
        ]
        
        for pattern in passive_patterns:
            if re.search(pattern, sentence.lower()):
                return True
        return False
    
    def _count_technical_terms(self, sentence: str) -> int:
        """Count technical terms in sentence."""
        sentence_lower = sentence.lower()
        count = 0
        
        # Count neuroscience terminology
        for term in self.neuro_terminology:
            if term in sentence_lower:
                count += 1
        
        # Count technical indicators
        for indicator in self.technical_indicators:
            if indicator in sentence_lower:
                count += 1
        
        return count
    
    def _calculate_readability(self, sentence: str, words: List[str]) -> float:
        """Calculate readability score (simplified Flesch approximation)."""
        if not words:
            return 0.0
        
        word_count = len(words)
        sentence_count = 1  # Single sentence
        syllable_count = sum(self._count_syllables(word) for word in words)
        
        # Simplified Flesch formula
        if word_count > 0 and syllable_count > 0:
            score = 206.835 - (1.015 * word_count) - (84.6 * syllable_count / word_count)
            # Normalize to 0-1 range
            return max(0.0, min(score / 100.0, 1.0))
        
        return 0.5  # Default middle score
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)."""
        word = word.lower().strip(string.punctuation)
        if not word:
            return 0
        
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(syllable_count, 1)
    
    def _extract_grammatical_features(self, sentence: str, words: List[str]) -> Dict[str, any]:
        """Extract grammatical features from sentence."""
        return {
            'word_count': len(words),
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
            'punctuation_marks': len(re.findall(r'[^\w\s]', sentence)),
            'capitalized_words': len([w for w in words if w.istitle()]),
            'has_question_mark': '?' in sentence,
            'has_exclamation': '!' in sentence,
            'parenthetical_info': len(re.findall(r'\([^)]*\)', sentence)),
            'numeric_values': len(re.findall(r'\b\d+\.?\d*\b', sentence))
        }
    
    def _extract_terminology(self, text: str) -> List[NeuroTerm]:
        """Extract neuroscience terminology from text."""
        terms = []
        text_lower = text.lower()
        
        for term, info in self.neuro_terminology.items():
            # Find all occurrences
            for match in re.finditer(re.escape(term), text_lower):
                start, end = match.span()
                context_start = max(0, start - 50)
                context_end = min(len(text), end + 50)
                context = text[context_start:context_end].strip()
                
                # Calculate confidence based on context
                confidence = self._calculate_term_confidence(term, context, info)
                
                neuro_term = NeuroTerm(
                    term=term,
                    category=info['category'],
                    context=context,
                    confidence=confidence,
                    position=(start, end)
                )
                terms.append(neuro_term)
        
        # Remove overlapping terms (keep higher confidence ones)
        terms = self._remove_overlapping_terms(terms)
        
        return terms
    
    def _calculate_term_confidence(self, term: str, context: str, info: Dict) -> float:
        """Calculate confidence score for terminology extraction."""
        base_confidence = 0.8
        
        # Boost confidence if term appears in formal context
        formal_indicators = ['study', 'research', 'analysis', 'investigation', 'examination']
        if any(indicator in context.lower() for indicator in formal_indicators):
            base_confidence += 0.1
        
        # Reduce confidence if term appears in very short context
        if len(context.split()) < 5:
            base_confidence -= 0.2
        
        return max(0.1, min(base_confidence, 1.0))
    
    def _remove_overlapping_terms(self, terms: List[NeuroTerm]) -> List[NeuroTerm]:
        """Remove overlapping term extractions, keeping higher confidence ones."""
        if not terms:
            return terms
        
        # Sort by position
        terms.sort(key=lambda t: t.position[0])
        
        filtered_terms = []
        for term in terms:
            # Check for overlap with already accepted terms
            overlaps = False
            for accepted_term in filtered_terms:
                if (term.position[0] < accepted_term.position[1] and 
                    term.position[1] > accepted_term.position[0]):
                    # There's overlap - keep the higher confidence term
                    if term.confidence > accepted_term.confidence:
                        filtered_terms.remove(accepted_term)
                        filtered_terms.append(term)
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_terms.append(term)
        
        return filtered_terms
    
    def _detect_writing_patterns(self, text: str, sentences: List[SentenceStructure]) -> List[WritingPattern]:
        """Detect common writing patterns in neuroscience text."""
        patterns = []
        
        # Pattern 1: Hypothesis-Method-Result-Conclusion structure
        has_hypothesis = any(s.sentence_type == SentenceType.HYPOTHESIS for s in sentences)
        has_method = any(s.sentence_type == SentenceType.METHOD_DESCRIPTION for s in sentences)
        has_result = any(s.sentence_type == SentenceType.RESULT_STATEMENT for s in sentences)
        has_conclusion = any(s.sentence_type == SentenceType.CONCLUSION for s in sentences)
        
        if has_hypothesis and has_method and has_result:
            patterns.append(WritingPattern(
                pattern_type="hypothesis_driven_structure",
                description="Follows hypothesis-method-result-conclusion structure",
                frequency=0.8,
                examples=[],
                confidence=0.9,
                section_types=["introduction", "methods", "results", "discussion"],
                linguistic_features={"structured_approach": True}
            ))
        
        # Pattern 2: Heavy citation integration
        citation_density = sum(s.citation_count for s in sentences) / len(sentences) if sentences else 0
        if citation_density > 0.3:  # More than 0.3 citations per sentence
            patterns.append(WritingPattern(
                pattern_type="citation_heavy",
                description="Extensive use of citations for support",
                frequency=citation_density,
                examples=[],
                confidence=0.8,
                section_types=["introduction", "discussion"],
                linguistic_features={"citation_density": citation_density}
            ))
        
        # Pattern 3: Technical terminology density
        technical_density = sum(s.technical_term_count for s in sentences) / sum(s.word_count for s in sentences) if sentences else 0
        if technical_density > 0.15:  # More than 15% technical terms
            patterns.append(WritingPattern(
                pattern_type="high_technical_density",
                description="High density of technical terminology",
                frequency=technical_density,
                examples=[],
                confidence=0.7,
                section_types=["methods", "results"],
                linguistic_features={"technical_density": technical_density}
            ))
        
        return patterns
    
    def _analyze_style(self, sentences: List[SentenceStructure], 
                      neuro_terms: List[NeuroTerm]) -> StyleCharacteristics:
        """Analyze overall writing style characteristics."""
        if not sentences:
            return StyleCharacteristics(
                formality_level=0.5, technical_density=0.0, sentence_complexity=0.5,
                citation_density=0.0, passive_voice_ratio=0.0, avg_sentence_length=0.0,
                vocabulary_richness=0.5, writing_style=WritingStyle.FORMAL_ACADEMIC,
                consistency_score=0.5
            )
        
        # Calculate metrics
        total_words = sum(s.word_count for s in sentences)
        total_citations = sum(s.citation_count for s in sentences)
        passive_sentences = sum(1 for s in sentences if s.passive_voice)
        total_technical_terms = sum(s.technical_term_count for s in sentences)
        avg_complexity = sum(s.complexity_score for s in sentences) / len(sentences)
        avg_sentence_length = total_words / len(sentences)
        
        # Calculate ratios
        citation_density = total_citations / len(sentences)
        passive_voice_ratio = passive_sentences / len(sentences)
        technical_density = total_technical_terms / total_words if total_words > 0 else 0
        
        # Estimate formality level
        formality_level = min((technical_density * 2 + passive_voice_ratio + citation_density) / 3, 1.0)
        
        # Calculate vocabulary richness (simplified)
        all_words = []
        for sentence in sentences:
            all_words.extend(sentence.text.lower().split())
        
        unique_words = len(set(all_words))
        total_word_instances = len(all_words)
        vocabulary_richness = unique_words / total_word_instances if total_word_instances > 0 else 0
        
        # Determine writing style
        if technical_density > 0.2:
            writing_style = WritingStyle.TECHNICAL_DESCRIPTIVE
        elif citation_density > 0.4:
            writing_style = WritingStyle.ANALYTICAL
        elif passive_voice_ratio > 0.6:
            writing_style = WritingStyle.FORMAL_ACADEMIC
        else:
            writing_style = WritingStyle.EXPLANATORY
        
        # Calculate consistency score
        complexity_variance = sum((s.complexity_score - avg_complexity) ** 2 for s in sentences) / len(sentences)
        consistency_score = max(0, 1 - complexity_variance)
        
        return StyleCharacteristics(
            formality_level=formality_level,
            technical_density=technical_density,
            sentence_complexity=avg_complexity,
            citation_density=citation_density,
            passive_voice_ratio=passive_voice_ratio,
            avg_sentence_length=avg_sentence_length,
            vocabulary_richness=vocabulary_richness,
            writing_style=writing_style,
            consistency_score=consistency_score
        )
    
    def _analyze_terminology_usage(self, neuro_terms: List[NeuroTerm], 
                                  total_words: int) -> TerminologyAnalysis:
        """Analyze neuroscience terminology usage patterns."""
        if not neuro_terms:
            return TerminologyAnalysis(
                total_terms=0, unique_terms=0, term_categories={},
                most_frequent_terms=[], terminology_density=0.0,
                category_distribution={}, complexity_level="basic"
            )
        
        # Count categories
        category_counts = Counter(term.category for term in neuro_terms)
        
        # Calculate category distribution
        total_terms = len(neuro_terms)
        category_distribution = {
            category: count / total_terms 
            for category, count in category_counts.items()
        }
        
        # Find most frequent terms
        term_frequencies = Counter(term.term for term in neuro_terms)
        most_frequent = [
            term for term in neuro_terms 
            if term_frequencies[term.term] >= max(term_frequencies.values())
        ][:10]
        
        # Calculate terminology density
        terminology_density = total_terms / total_words if total_words > 0 else 0
        
        # Determine complexity level
        complexity_scores = {'basic': 1, 'intermediate': 2, 'advanced': 3}
        avg_complexity = sum(
            complexity_scores.get(self.neuro_terminology.get(term.term, {}).get('complexity', 'basic'), 1)
            for term in neuro_terms
        ) / total_terms if total_terms > 0 else 1
        
        if avg_complexity < 1.5:
            complexity_level = "basic"
        elif avg_complexity < 2.5:
            complexity_level = "intermediate"
        else:
            complexity_level = "advanced"
        
        unique_terms = len(set(term.term for term in neuro_terms))
        
        return TerminologyAnalysis(
            total_terms=total_terms,
            unique_terms=unique_terms,
            term_categories=dict(category_counts),
            most_frequent_terms=most_frequent,
            terminology_density=terminology_density,
            category_distribution=category_distribution,
            complexity_level=complexity_level
        )
    
    def _calculate_coherence(self, text: str, 
                           sentences: List[SentenceStructure]) -> CoherenceMetrics:
        """Calculate text coherence metrics."""
        # Simplified coherence calculation
        # In a full implementation, this would use advanced NLP techniques
        
        # Lexical cohesion - repetition of key terms
        words = text.lower().split()
        word_freq = Counter(words)
        repeated_words = sum(1 for count in word_freq.values() if count > 1)
        lexical_cohesion = min(repeated_words / len(word_freq) if word_freq else 0, 1.0)
        
        # Transition quality - presence of transition words
        transition_count = sum(
            1 for word in words 
            if any(trans in ' '.join(words[i:i+3]) for trans in self.transition_words for i in range(len(words)-2))
        )
        transition_quality = min(transition_count / len(sentences) if sentences else 0, 1.0)
        
        # Logical flow - sentence type progression
        sentence_types = [s.sentence_type for s in sentences]
        logical_patterns = [
            [SentenceType.BACKGROUND, SentenceType.OBJECTIVE, SentenceType.METHOD_DESCRIPTION],
            [SentenceType.HYPOTHESIS, SentenceType.RESULT_STATEMENT, SentenceType.CONCLUSION],
            [SentenceType.CITATION_CONTEXT, SentenceType.OBJECTIVE]
        ]
        
        logical_flow_score = 0.5  # Default middle score
        for pattern in logical_patterns:
            if all(stype in sentence_types for stype in pattern):
                logical_flow_score = min(logical_flow_score + 0.2, 1.0)
        
        # Topic consistency - based on terminology repetition
        term_consistency = len(set(term.term for term in sentences if hasattr(sentences, 'neuro_terms')))
        topic_consistency = min(term_consistency / 10.0, 1.0) if term_consistency else 0.5
        
        # Semantic similarity (simplified - based on word overlap)
        semantic_similarity = lexical_cohesion  # Simplified approximation
        
        return CoherenceMetrics(
            lexical_cohesion=lexical_cohesion,
            semantic_similarity=semantic_similarity,
            transition_quality=transition_quality,
            logical_flow_score=logical_flow_score,
            topic_consistency=topic_consistency,
            overall_coherence=0.0  # Will be calculated in __post_init__
        )
    
    def _calculate_quality_score(self, style: StyleCharacteristics, 
                               terminology: TerminologyAnalysis,
                               coherence: CoherenceMetrics) -> float:
        """Calculate overall text quality score."""
        # Weight different aspects of quality
        weights = {
            'style_consistency': 0.2,
            'technical_appropriateness': 0.25,
            'terminology_usage': 0.25,
            'coherence': 0.3
        }
        
        scores = {
            'style_consistency': style.consistency_score,
            'technical_appropriateness': min(style.technical_density * 2, 1.0),
            'terminology_usage': min(terminology.terminology_density * 5, 1.0),
            'coherence': coherence.overall_coherence
        }
        
        quality_score = sum(scores[aspect] * weight for aspect, weight in weights.items())
        return min(quality_score, 1.0)