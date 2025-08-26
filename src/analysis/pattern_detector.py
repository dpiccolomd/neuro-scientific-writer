"""Writing pattern detection for neuroscience literature."""

import re
import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter, defaultdict

from .models import WritingPattern, AnalysisResult, SentenceType
from .exceptions import PatternDetectionError

logger = logging.getLogger(__name__)


class WritingPatternDetector:
    """Detects and analyzes writing patterns in neuroscience literature."""
    
    def __init__(self):
        """Initialize pattern detector with neuroscience-specific patterns."""
        self.introduction_patterns = self._load_introduction_patterns()
        self.argument_patterns = self._load_argument_patterns()
        self.methodology_patterns = self._load_methodology_patterns()
        self.result_patterns = self._load_result_patterns()
        self.discussion_patterns = self._load_discussion_patterns()
        
    def detect_patterns(self, analysis_results: List[AnalysisResult]) -> List[WritingPattern]:
        """
        Detect writing patterns across multiple analysis results.
        
        Args:
            analysis_results: List of text analysis results
            
        Returns:
            List of detected writing patterns
            
        Raises:
            PatternDetectionError: If pattern detection fails
        """
        if not analysis_results:
            raise PatternDetectionError("No analysis results provided")
        
        try:
            patterns = []
            
            # Detect structural patterns
            structural_patterns = self._detect_structural_patterns(analysis_results)
            patterns.extend(structural_patterns)
            
            # Detect linguistic patterns
            linguistic_patterns = self._detect_linguistic_patterns(analysis_results)
            patterns.extend(linguistic_patterns)
            
            # Detect citation patterns
            citation_patterns = self._detect_citation_patterns(analysis_results)
            patterns.extend(citation_patterns)
            
            # Detect terminology patterns
            terminology_patterns = self._detect_terminology_patterns(analysis_results)
            patterns.extend(terminology_patterns)
            
            # Detect argument structure patterns
            argument_patterns = self._detect_argument_patterns(analysis_results)
            patterns.extend(argument_patterns)
            
            logger.info(f"Detected {len(patterns)} writing patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            raise PatternDetectionError(f"Failed to detect patterns: {e}")
    
    def _load_introduction_patterns(self) -> Dict[str, List[str]]:
        """Load common introduction patterns in neuroscience."""
        return {
            'funnel_structure': [
                r'(?:the\s+)?(?:human\s+)?brain\s+is',
                r'(?:understanding|knowledge)\s+(?:of|about)',
                r'however,?\s+(?:little|limited|less)\s+is\s+known',
                r'(?:the\s+)?(?:present|current)\s+study'
            ],
            'gap_identification': [
                r'(?:however|nevertheless|despite),?\s+(?:little|limited|few)',
                r'(?:remains?|is|are)\s+(?:unclear|unknown|poorly understood)',
                r'(?:no|few)\s+(?:studies|research|investigations?)',
                r'(?:little|limited)\s+(?:attention|focus|research)'
            ],
            'significance_statement': [
                r'(?:understanding|investigating|examining)\s+(?:this|these)',
                r'(?:this|these)\s+(?:findings?|results?|insights?)\s+(?:may|could|would)',
                r'(?:important|crucial|critical)\s+(?:implications?|consequences?)',
                r'(?:clinical|therapeutic)\s+(?:relevance|implications?|significance)'
            ]
        }
    
    def _load_argument_patterns(self) -> Dict[str, List[str]]:
        """Load argumentation patterns."""
        return {
            'evidence_presentation': [
                r'(?:evidence|data|findings?|results?)\s+(?:suggest|indicate|show|demonstrate)',
                r'(?:consistent|in line)\s+with\s+(?:previous|prior)',
                r'(?:support|supports?)\s+(?:the\s+)?(?:hypothesis|notion|idea)',
                r'(?:contrary|in contrast)\s+to\s+(?:previous|prior)'
            ],
            'causal_relationships': [
                r'(?:due\s+to|because\s+of|as\s+a\s+result\s+of)',
                r'(?:leads?\s+to|results?\s+in|causes?)',
                r'(?:consequently|therefore|thus|hence)',
                r'(?:underlying|mediating|moderating)\s+(?:mechanism|factor)'
            ],
            'comparison_contrast': [
                r'(?:compared\s+to|in\s+comparison\s+to|relative\s+to)',
                r'(?:whereas|while|in\s+contrast)',
                r'(?:similarly|likewise|in\s+the\s+same\s+way)',
                r'(?:different|distinct|unique)\s+from'
            ]
        }
    
    def _load_methodology_patterns(self) -> Dict[str, List[str]]:
        """Load methodology description patterns."""
        return {
            'participant_description': [
                r'(?:\d+|all)\s+(?:participants?|subjects?|patients?)',
                r'(?:healthy|control)\s+(?:participants?|subjects?|volunteers?)',
                r'(?:inclusion|exclusion)\s+criteria',
                r'(?:informed\s+)?consent\s+(?:was\s+)?(?:obtained|provided)'
            ],
            'procedure_description': [
                r'(?:participants?|subjects?)\s+(?:were|underwent)',
                r'(?:the\s+)?(?:experiment|study|task)\s+(?:consisted|comprised)',
                r'(?:each|every)\s+(?:trial|session|block)',
                r'(?:randomized|counterbalanced|controlled)'
            ],
            'analysis_description': [
                r'(?:data|results?)\s+(?:were|was)\s+(?:analyzed|processed)',
                r'(?:statistical|data)\s+analysis',
                r'(?:using|employing|utilizing)\s+(?:spss|r|matlab|python)',
                r'(?:significance|alpha)\s+level\s+(?:was\s+)?set'
            ]
        }
    
    def _load_result_patterns(self) -> Dict[str, List[str]]:
        """Load result presentation patterns."""
        return {
            'statistical_reporting': [
                r'(?:significant|non-significant)\s+(?:difference|effect|correlation)',
                r'[pt]\s*[<>=]\s*[0-9.]+',
                r'(?:f|t|χ²|z)\s*\([^)]+\)\s*=\s*[0-9.]+',
                r'(?:confidence\s+interval|ci)\s*[=:]?\s*\[?[0-9.,\s-]+\]?'
            ],
            'descriptive_reporting': [
                r'(?:increased|decreased|higher|lower)\s+(?:in|for|among)',
                r'(?:showed?|demonstrated|exhibited|displayed)',
                r'(?:positive|negative|strong|weak)\s+(?:correlation|relationship|association)',
                r'(?:main|interaction)\s+effect\s+(?:of|for)'
            ]
        }
    
    def _load_discussion_patterns(self) -> Dict[str, List[str]]:
        """Load discussion section patterns."""
        return {
            'interpretation': [
                r'(?:these|our)\s+(?:findings?|results?)\s+(?:suggest|indicate|demonstrate)',
                r'(?:consistent|in line|agreement)\s+with\s+(?:previous|prior)',
                r'(?:interpretation|explanation)\s+(?:is|could\s+be)',
                r'(?:alternative|another)\s+(?:explanation|interpretation|possibility)'
            ],
            'limitations': [
                r'(?:limitation|caveat|constraint)\s+(?:of|in)',
                r'(?:should\s+be\s+)?(?:interpreted|considered)\s+(?:with\s+)?caution',
                r'(?:future\s+)?(?:research|studies?)\s+(?:should|need|ought)',
                r'(?:cannot|unable\s+to)\s+(?:rule\s+out|exclude|determine)'
            ],
            'implications': [
                r'(?:clinical|therapeutic|practical)\s+(?:implications?|significance)',
                r'(?:important|significant)\s+(?:for|to)\s+(?:understanding|treatment)',
                r'(?:may\s+have\s+)?implications?\s+for',
                r'(?:contribute|adds?)\s+to\s+(?:our\s+)?(?:understanding|knowledge)'
            ]
        }
    
    def _detect_structural_patterns(self, results: List[AnalysisResult]) -> List[WritingPattern]:
        """Detect document structure patterns."""
        patterns = []
        
        # Analyze sentence type distributions
        all_sentence_types = []
        for result in results:
            all_sentence_types.extend([s.sentence_type for s in result.sentences])
        
        type_counts = Counter(all_sentence_types)
        total_sentences = len(all_sentence_types)
        
        if total_sentences == 0:
            return patterns
        
        # Check for hypothesis-driven structure
        if (type_counts[SentenceType.HYPOTHESIS] / total_sentences > 0.05 and
            type_counts[SentenceType.RESULT_STATEMENT] / total_sentences > 0.1):
            
            patterns.append(WritingPattern(
                pattern_type="hypothesis_driven_structure",
                description="Follows hypothesis-testing framework with clear predictions and results",
                frequency=type_counts[SentenceType.HYPOTHESIS] / total_sentences,
                examples=[],
                confidence=0.8,
                section_types=["introduction", "results"],
                linguistic_features={
                    "hypothesis_density": type_counts[SentenceType.HYPOTHESIS] / total_sentences,
                    "result_density": type_counts[SentenceType.RESULT_STATEMENT] / total_sentences
                }
            ))
        
        # Check for method-heavy structure
        if type_counts[SentenceType.METHOD_DESCRIPTION] / total_sentences > 0.15:
            patterns.append(WritingPattern(
                pattern_type="method_detailed_structure",
                description="Extensive methodological descriptions with detailed procedures",
                frequency=type_counts[SentenceType.METHOD_DESCRIPTION] / total_sentences,
                examples=[],
                confidence=0.7,
                section_types=["methods"],
                linguistic_features={
                    "method_density": type_counts[SentenceType.METHOD_DESCRIPTION] / total_sentences
                }
            ))
        
        return patterns
    
    def _detect_linguistic_patterns(self, results: List[AnalysisResult]) -> List[WritingPattern]:
        """Detect linguistic and stylistic patterns."""
        patterns = []
        
        # Aggregate style characteristics
        all_styles = [result.style_characteristics for result in results]
        
        if not all_styles:
            return patterns
        
        # Calculate averages
        avg_formality = sum(s.formality_level for s in all_styles) / len(all_styles)
        avg_technical_density = sum(s.technical_density for s in all_styles) / len(all_styles)
        avg_passive_voice = sum(s.passive_voice_ratio for s in all_styles) / len(all_styles)
        avg_sentence_length = sum(s.avg_sentence_length for s in all_styles) / len(all_styles)
        
        # High formality pattern
        if avg_formality > 0.7:
            patterns.append(WritingPattern(
                pattern_type="high_formality",
                description="Consistently formal academic writing style",
                frequency=avg_formality,
                examples=[],
                confidence=0.8,
                section_types=["all"],
                linguistic_features={
                    "formality_level": avg_formality,
                    "technical_density": avg_technical_density,
                    "passive_voice_ratio": avg_passive_voice
                }
            ))
        
        # Technical density pattern
        if avg_technical_density > 0.2:
            patterns.append(WritingPattern(
                pattern_type="high_technical_density",
                description="Heavy use of technical and specialized terminology",
                frequency=avg_technical_density,
                examples=[],
                confidence=0.9,
                section_types=["methods", "results"],
                linguistic_features={
                    "technical_density": avg_technical_density,
                    "terminology_complexity": "high"
                }
            ))
        
        # Complex sentence structure pattern
        if avg_sentence_length > 20:
            patterns.append(WritingPattern(
                pattern_type="complex_sentences",
                description="Preference for long, complex sentence structures",
                frequency=min(avg_sentence_length / 30, 1.0),
                examples=[],
                confidence=0.7,
                section_types=["introduction", "discussion"],
                linguistic_features={
                    "avg_sentence_length": avg_sentence_length,
                    "complexity_preference": "high"
                }
            ))
        
        return patterns
    
    def _detect_citation_patterns(self, results: List[AnalysisResult]) -> List[WritingPattern]:
        """Detect citation usage patterns."""
        patterns = []
        
        # Calculate citation statistics
        total_sentences = sum(len(result.sentences) for result in results)
        total_citations = sum(
            sum(s.citation_count for s in result.sentences) 
            for result in results
        )
        
        if total_sentences == 0:
            return patterns
        
        citation_density = total_citations / total_sentences
        
        # High citation density pattern
        if citation_density > 0.4:
            patterns.append(WritingPattern(
                pattern_type="citation_heavy",
                description="Extensive citation usage for supporting arguments",
                frequency=citation_density,
                examples=[],
                confidence=0.8,
                section_types=["introduction", "discussion"],
                linguistic_features={
                    "citation_density": citation_density,
                    "evidence_based_approach": True
                }
            ))
        
        # Analyze citation contexts (simplified)
        citation_sentences = []
        for result in results:
            for sentence in result.sentences:
                if sentence.citation_count > 0:
                    citation_sentences.append(sentence)
        
        if citation_sentences:
            citation_context_types = Counter(s.sentence_type for s in citation_sentences)
            
            # Background-heavy citations
            if citation_context_types[SentenceType.BACKGROUND] > len(citation_sentences) * 0.4:
                patterns.append(WritingPattern(
                    pattern_type="background_citation_pattern",
                    description="Citations primarily used for background information",
                    frequency=citation_context_types[SentenceType.BACKGROUND] / len(citation_sentences),
                    examples=[],
                    confidence=0.7,
                    section_types=["introduction"],
                    linguistic_features={
                        "background_citation_ratio": citation_context_types[SentenceType.BACKGROUND] / len(citation_sentences)
                    }
                ))
        
        return patterns
    
    def _detect_terminology_patterns(self, results: List[AnalysisResult]) -> List[WritingPattern]:
        """Detect neuroscience terminology usage patterns."""
        patterns = []
        
        # Aggregate terminology data
        all_terms = []
        for result in results:
            all_terms.extend(result.neuro_terms)
        
        if not all_terms:
            return patterns
        
        # Analyze category distribution
        category_counts = Counter(term.category for term in all_terms)
        total_terms = len(all_terms)
        
        # Anatomy-focused pattern
        if category_counts.get('anatomy', 0) / total_terms > 0.4:
            patterns.append(WritingPattern(
                pattern_type="anatomy_focused",
                description="Heavy emphasis on anatomical terminology and structures",
                frequency=category_counts['anatomy'] / total_terms,
                examples=[],
                confidence=0.8,
                section_types=["introduction", "methods"],
                linguistic_features={
                    "anatomy_ratio": category_counts['anatomy'] / total_terms,
                    "structural_emphasis": True
                }
            ))
        
        # Function-focused pattern
        if category_counts.get('function', 0) / total_terms > 0.3:
            patterns.append(WritingPattern(
                pattern_type="function_focused",
                description="Emphasis on functional processes and mechanisms",
                frequency=category_counts['function'] / total_terms,
                examples=[],
                confidence=0.8,
                section_types=["introduction", "discussion"],
                linguistic_features={
                    "function_ratio": category_counts['function'] / total_terms,
                    "mechanistic_focus": True
                }
            ))
        
        # Technique-heavy pattern
        if category_counts.get('technique', 0) / total_terms > 0.25:
            patterns.append(WritingPattern(
                pattern_type="technique_heavy",
                description="Extensive discussion of research methods and techniques",
                frequency=category_counts['technique'] / total_terms,
                examples=[],
                confidence=0.9,
                section_types=["methods"],
                linguistic_features={
                    "technique_ratio": category_counts['technique'] / total_terms,
                    "methodological_focus": True
                }
            ))
        
        return patterns
    
    def _detect_argument_patterns(self, results: List[AnalysisResult]) -> List[WritingPattern]:
        """Detect argumentation and reasoning patterns."""
        patterns = []
        
        # Combine all text for pattern matching
        all_text = []
        for result in results:
            for sentence in result.sentences:
                all_text.append(sentence.text.lower())
        
        combined_text = ' '.join(all_text)
        
        # Check for different argument patterns
        for pattern_category, pattern_list in self.argument_patterns.items():
            matches = 0
            total_patterns = len(pattern_list)
            
            for pattern in pattern_list:
                if re.search(pattern, combined_text):
                    matches += 1
            
            if matches / total_patterns > 0.3:  # At least 30% of patterns found
                confidence = min(matches / total_patterns, 1.0)
                
                patterns.append(WritingPattern(
                    pattern_type=f"argument_{pattern_category}",
                    description=f"Strong use of {pattern_category.replace('_', ' ')} argumentative strategies",
                    frequency=confidence,
                    examples=[],
                    confidence=confidence,
                    section_types=["introduction", "discussion"],
                    linguistic_features={
                        "argument_type": pattern_category,
                        "pattern_coverage": matches / total_patterns
                    }
                ))
        
        return patterns
    
    def generate_template_from_patterns(self, patterns: List[WritingPattern]) -> Dict[str, any]:
        """Generate a writing template based on detected patterns."""
        template = {
            "structure": {},
            "style_guidelines": {},
            "linguistic_features": {},
            "pattern_summary": {}
        }
        
        # Analyze structural patterns
        structural_patterns = [p for p in patterns if 'structure' in p.pattern_type]
        if structural_patterns:
            template["structure"]["recommended_approach"] = [
                p.description for p in structural_patterns
            ]
        
        # Extract style guidelines
        style_patterns = [p for p in patterns if p.pattern_type in 
                         ['high_formality', 'high_technical_density', 'complex_sentences']]
        
        for pattern in style_patterns:
            template["style_guidelines"][pattern.pattern_type] = {
                "description": pattern.description,
                "frequency": pattern.frequency,
                "sections": pattern.section_types
            }
        
        # Extract linguistic features
        for pattern in patterns:
            template["linguistic_features"][pattern.pattern_type] = pattern.linguistic_features
        
        # Generate pattern summary
        template["pattern_summary"] = {
            "total_patterns": len(patterns),
            "high_confidence_patterns": len([p for p in patterns if p.confidence > 0.8]),
            "primary_focus": self._identify_primary_focus(patterns),
            "writing_complexity": self._assess_writing_complexity(patterns)
        }
        
        return template
    
    def _identify_primary_focus(self, patterns: List[WritingPattern]) -> str:
        """Identify the primary focus based on detected patterns."""
        focus_indicators = {
            'anatomy_focused': 'anatomical',
            'function_focused': 'functional',
            'technique_heavy': 'methodological',
            'hypothesis_driven_structure': 'experimental',
            'citation_heavy': 'literature_based'
        }
        
        for pattern in sorted(patterns, key=lambda p: p.confidence, reverse=True):
            if pattern.pattern_type in focus_indicators:
                return focus_indicators[pattern.pattern_type]
        
        return 'general_neuroscience'
    
    def _assess_writing_complexity(self, patterns: List[WritingPattern]) -> str:
        """Assess overall writing complexity."""
        complexity_indicators = ['high_technical_density', 'complex_sentences', 'high_formality']
        complexity_count = sum(1 for p in patterns if p.pattern_type in complexity_indicators)
        
        if complexity_count >= 3:
            return 'high'
        elif complexity_count >= 2:
            return 'medium'
        else:
            return 'basic'