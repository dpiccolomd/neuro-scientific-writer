"""Empirical pattern detection system based on statistical analysis of published papers."""

import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .models import AnalysisResult, WritingPattern, SentenceType
from .exceptions import PatternDetectionError
from pdf_processor.models import ProcessedDocument

logger = logging.getLogger(__name__)


@dataclass
class EmpiricalPattern:
    """Empirically derived writing pattern."""
    pattern_id: str
    pattern_type: str
    description: str
    statistical_evidence: Dict[str, float]
    sample_size: int
    confidence_interval: Tuple[float, float]
    journals_analyzed: List[str]
    domains_analyzed: List[str]
    validation_score: float
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def is_statistically_significant(self) -> bool:
        """Check if pattern has statistical significance."""
        return (
            self.sample_size >= 50 and 
            self.validation_score >= 0.7 and
            len(self.journals_analyzed) >= 3
        )


@dataclass
class StructuralMetrics:
    """Structural analysis metrics for an introduction."""
    document_id: str
    total_paragraphs: int
    total_sentences: int
    avg_sentences_per_paragraph: float
    paragraph_lengths: List[int]  # sentences per paragraph
    conceptual_breadth_progression: List[float]  # 0-1 scale, broad to specific
    argumentation_structure: str  # problem_gap_solution, hypothesis_test, etc.
    transition_sophistication: float  # 0-1 scale
    information_density: List[float]  # concepts per sentence by paragraph
    citation_distribution: List[int]  # citations per paragraph
    journal: str
    domain: str
    year: int
    impact_factor: Optional[float] = None


@dataclass
class DataCollectionResult:
    """Result of empirical data collection."""
    total_papers_analyzed: int
    successful_extractions: int
    failed_extractions: int
    journals_covered: List[str]
    domains_covered: List[str]
    structural_metrics: List[StructuralMetrics]
    patterns_identified: List[EmpiricalPattern]
    collection_date: datetime = field(default_factory=datetime.now)


class EmpiricalPatternDetector:
    """
    Empirically-driven pattern detection system.
    
    This system replaces naive rule-based patterns with statistically derived
    patterns based on analysis of actual published literature.
    """
    
    def __init__(self, data_dir: str = "data/empirical_patterns"):
        """Initialize empirical pattern detector."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.structural_patterns = []
        self.argumentative_patterns = []
        self.transitional_patterns = []
        self.domain_specific_patterns = {}
        self.journal_specific_patterns = {}
        
        # Load existing patterns
        self._load_existing_patterns()
        
        # Neuroscience domain vocabulary
        self.domain_indicators = self._load_domain_indicators()
        
    def collect_empirical_data(
        self, 
        documents: List[ProcessedDocument],
        min_sample_size: int = 50
    ) -> DataCollectionResult:
        """
        Collect empirical data from a set of published papers.
        
        Args:
            documents: List of processed research papers
            min_sample_size: Minimum sample size for pattern validation
            
        Returns:
            DataCollectionResult with collected metrics and patterns
            
        Raises:
            PatternDetectionError: If data collection fails
        """
        if len(documents) < min_sample_size:
            raise PatternDetectionError(
                f"Insufficient sample size: {len(documents)} < {min_sample_size}"
            )
        
        try:
            logger.info(f"Starting empirical data collection from {len(documents)} papers")
            
            structural_metrics = []
            successful = 0
            failed = 0
            
            journals = set()
            domains = set()
            
            for doc in documents:
                try:
                    if doc.introduction_section:
                        metrics = self._analyze_document_structure(doc)
                        if metrics:
                            structural_metrics.append(metrics)
                            journals.add(metrics.journal)
                            domains.add(metrics.domain)
                            successful += 1
                        else:
                            failed += 1
                    else:
                        failed += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to analyze document {doc.document_id}: {e}")
                    failed += 1
            
            # Identify patterns from collected data
            patterns = self._identify_empirical_patterns(structural_metrics)
            
            result = DataCollectionResult(
                total_papers_analyzed=len(documents),
                successful_extractions=successful,
                failed_extractions=failed,
                journals_covered=list(journals),
                domains_covered=list(domains),
                structural_metrics=structural_metrics,
                patterns_identified=patterns
            )
            
            # Save results
            self._save_collection_results(result)
            
            logger.info(f"Data collection completed: {successful} successful, {len(patterns)} patterns identified")
            return result
            
        except Exception as e:
            logger.error(f"Empirical data collection failed: {e}")
            raise PatternDetectionError(f"Data collection failed: {e}")
    
    def detect_patterns_empirical(
        self, 
        analysis_results: List[AnalysisResult],
        require_statistical_significance: bool = True
    ) -> List[EmpiricalPattern]:
        """
        Detect patterns using empirically derived methods.
        
        Args:
            analysis_results: Analysis results to evaluate
            require_statistical_significance: Only return statistically significant patterns
            
        Returns:
            List of empirically validated patterns
        """
        try:
            detected_patterns = []
            
            # Analyze structural patterns
            structural = self._detect_empirical_structural_patterns(analysis_results)
            detected_patterns.extend(structural)
            
            # Analyze argumentative patterns
            argumentative = self._detect_empirical_argumentative_patterns(analysis_results)
            detected_patterns.extend(argumentative)
            
            # Analyze transitional patterns
            transitional = self._detect_empirical_transitional_patterns(analysis_results)
            detected_patterns.extend(transitional)
            
            # Filter by statistical significance if required
            if require_statistical_significance:
                detected_patterns = [p for p in detected_patterns if p.is_statistically_significant]
            
            logger.info(f"Detected {len(detected_patterns)} empirical patterns")
            return detected_patterns
            
        except Exception as e:
            logger.error(f"Empirical pattern detection failed: {e}")
            raise PatternDetectionError(f"Pattern detection failed: {e}")
    
    def _analyze_document_structure(self, doc: ProcessedDocument) -> Optional[StructuralMetrics]:
        """Analyze the structural characteristics of a document's introduction."""
        try:
            if not doc.introduction_section:
                return None
            
            intro_text = doc.introduction_section.content
            paragraphs = self._split_into_paragraphs(intro_text)
            
            if len(paragraphs) < 2:  # Need at least 2 paragraphs for analysis
                return None
            
            # Calculate basic structural metrics
            total_paragraphs = len(paragraphs)
            sentences_per_paragraph = []
            
            for para in paragraphs:
                sentences = self._split_into_sentences(para)
                sentences_per_paragraph.append(len(sentences))
            
            total_sentences = sum(sentences_per_paragraph)
            avg_sentences_per_paragraph = total_sentences / total_paragraphs
            
            # Analyze conceptual breadth progression
            breadth_progression = self._analyze_conceptual_breadth(paragraphs)
            
            # Determine argumentation structure
            arg_structure = self._classify_argumentation_structure(paragraphs)
            
            # Calculate transition sophistication
            transition_score = self._calculate_transition_sophistication(paragraphs)
            
            # Calculate information density
            info_density = self._calculate_information_density(paragraphs)
            
            # Count citations per paragraph
            citation_distribution = self._count_citations_per_paragraph(paragraphs)
            
            # Extract journal and domain information
            journal = self._extract_journal_name(doc)
            domain = self._classify_research_domain(intro_text)
            
            return StructuralMetrics(
                document_id=doc.document_id,
                total_paragraphs=total_paragraphs,
                total_sentences=total_sentences,
                avg_sentences_per_paragraph=avg_sentences_per_paragraph,
                paragraph_lengths=sentences_per_paragraph,
                conceptual_breadth_progression=breadth_progression,
                argumentation_structure=arg_structure,
                transition_sophistication=transition_score,
                information_density=info_density,
                citation_distribution=citation_distribution,
                journal=journal,
                domain=domain,
                year=doc.metadata.get('year', 2023)
            )
            
        except Exception as e:
            logger.warning(f"Failed to analyze document structure for {doc.document_id}: {e}")
            return None
    
    def _analyze_conceptual_breadth(self, paragraphs: List[str]) -> List[float]:
        """
        Analyze how conceptual breadth progresses from broad to specific.
        
        Returns scores 0-1 where 1 is most broad, 0 is most specific.
        """
        breadth_scores = []
        
        # Define broad vs specific indicators
        broad_indicators = [
            'field', 'domain', 'area', 'research', 'studies', 'literature',
            'understanding', 'knowledge', 'important', 'fundamental', 'critical'
        ]
        
        specific_indicators = [
            'specific', 'particular', 'this study', 'our', 'we', 'present',
            'investigate', 'examine', 'test', 'measure', 'analyze'
        ]
        
        for para in paragraphs:
            para_lower = para.lower()
            broad_count = sum(1 for word in broad_indicators if word in para_lower)
            specific_count = sum(1 for word in specific_indicators if word in para_lower)
            
            total_indicators = broad_count + specific_count
            if total_indicators == 0:
                breadth_score = 0.5  # Neutral
            else:
                breadth_score = broad_count / total_indicators
            
            breadth_scores.append(breadth_score)
        
        return breadth_scores
    
    def _classify_argumentation_structure(self, paragraphs: List[str]) -> str:
        """Classify the overall argumentation structure of the introduction."""
        combined_text = ' '.join(paragraphs).lower()
        
        # Problem-Gap-Solution structure indicators
        problem_indicators = ['problem', 'challenge', 'difficulty', 'limitation', 'issue']
        gap_indicators = ['gap', 'unknown', 'unclear', 'limited', 'lack', 'missing']
        solution_indicators = ['approach', 'method', 'technique', 'solution', 'address']
        
        # Hypothesis-Test structure indicators
        hypothesis_indicators = ['hypothesis', 'predict', 'expect', 'propose', 'suggest']
        test_indicators = ['test', 'examine', 'investigate', 'analyze', 'measure']
        
        # Count indicators
        pgs_score = (
            sum(1 for word in problem_indicators if word in combined_text) +
            sum(1 for word in gap_indicators if word in combined_text) +
            sum(1 for word in solution_indicators if word in combined_text)
        )
        
        ht_score = (
            sum(1 for word in hypothesis_indicators if word in combined_text) +
            sum(1 for word in test_indicators if word in combined_text)
        )
        
        if pgs_score > ht_score:
            return 'problem_gap_solution'
        elif ht_score > pgs_score:
            return 'hypothesis_test'
        else:
            return 'mixed'
    
    def _calculate_transition_sophistication(self, paragraphs: List[str]) -> float:
        """Calculate the sophistication of transitions between paragraphs."""
        if len(paragraphs) < 2:
            return 0.0
        
        sophisticated_transitions = [
            'furthermore', 'moreover', 'consequently', 'nevertheless', 'however',
            'in contrast', 'building on', 'extending', 'given that', 'despite'
        ]
        
        basic_transitions = [
            'also', 'and', 'but', 'so', 'then', 'next', 'first', 'second'
        ]
        
        transition_scores = []
        
        for i in range(1, len(paragraphs)):
            para_start = paragraphs[i][:100].lower()  # First 100 characters
            
            sophisticated_count = sum(1 for trans in sophisticated_transitions if trans in para_start)
            basic_count = sum(1 for trans in basic_transitions if trans in para_start)
            
            if sophisticated_count > 0:
                transition_scores.append(1.0)
            elif basic_count > 0:
                transition_scores.append(0.5)
            else:
                transition_scores.append(0.0)
        
        return np.mean(transition_scores) if transition_scores else 0.0
    
    def _calculate_information_density(self, paragraphs: List[str]) -> List[float]:
        """Calculate information density (concepts per sentence) for each paragraph."""
        density_scores = []
        
        # Neuroscience concept indicators
        concept_indicators = self.domain_indicators.get('neuroscience_terms', [])
        
        for para in paragraphs:
            sentences = self._split_into_sentences(para)
            if not sentences:
                density_scores.append(0.0)
                continue
            
            total_concepts = 0
            para_lower = para.lower()
            
            for concept in concept_indicators:
                total_concepts += para_lower.count(concept.lower())
            
            density = total_concepts / len(sentences)
            density_scores.append(density)
        
        return density_scores
    
    def _count_citations_per_paragraph(self, paragraphs: List[str]) -> List[int]:
        """Count citations in each paragraph."""
        citation_counts = []
        
        # Simple citation patterns
        citation_pattern = r'\([A-Za-z]+.*?\d{4}\)|\[\d+\]'
        
        for para in paragraphs:
            import re
            citations = re.findall(citation_pattern, para)
            citation_counts.append(len(citations))
        
        return citation_counts
    
    def _identify_empirical_patterns(
        self, 
        structural_metrics: List[StructuralMetrics]
    ) -> List[EmpiricalPattern]:
        """Identify patterns from collected structural metrics."""
        patterns = []
        
        if len(structural_metrics) < 20:  # Need minimum sample
            logger.warning(f"Insufficient data for pattern identification: {len(structural_metrics)}")
            return patterns
        
        # Convert to DataFrame for analysis
        df = self._metrics_to_dataframe(structural_metrics)
        
        # Identify structural patterns
        structural_patterns = self._identify_structural_patterns(df)
        patterns.extend(structural_patterns)
        
        # Identify argumentation patterns
        argument_patterns = self._identify_argument_patterns(df)
        patterns.extend(argument_patterns)
        
        # Identify journal-specific patterns (if enough data)
        journal_patterns = self._identify_journal_patterns(df)
        patterns.extend(journal_patterns)
        
        logger.info(f"Identified {len(patterns)} empirical patterns")
        return patterns
    
    def _metrics_to_dataframe(self, metrics: List[StructuralMetrics]) -> pd.DataFrame:
        """Convert structural metrics to pandas DataFrame for analysis."""
        data = []
        for m in metrics:
            row = {
                'document_id': m.document_id,
                'total_paragraphs': m.total_paragraphs,
                'total_sentences': m.total_sentences,
                'avg_sentences_per_paragraph': m.avg_sentences_per_paragraph,
                'argumentation_structure': m.argumentation_structure,
                'transition_sophistication': m.transition_sophistication,
                'journal': m.journal,
                'domain': m.domain,
                'year': m.year,
                'breadth_progression_slope': self._calculate_slope(m.conceptual_breadth_progression),
                'avg_info_density': np.mean(m.information_density) if m.information_density else 0,
                'total_citations': sum(m.citation_distribution)
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _identify_structural_patterns(self, df: pd.DataFrame) -> List[EmpiricalPattern]:
        """Identify statistically significant structural patterns."""
        patterns = []
        
        # Pattern 1: Optimal paragraph count distribution
        paragraph_counts = df['total_paragraphs'].values
        mean_paragraphs = np.mean(paragraph_counts)
        std_paragraphs = np.std(paragraph_counts)
        
        if std_paragraphs > 0:  # Avoid division by zero
            patterns.append(EmpiricalPattern(
                pattern_id="empirical_paragraph_count",
                pattern_type="structural",
                description=f"Optimal introduction length: {mean_paragraphs:.1f} ± {std_paragraphs:.1f} paragraphs",
                statistical_evidence={
                    'mean': mean_paragraphs,
                    'std': std_paragraphs,
                    'confidence_95': 1.96 * std_paragraphs / np.sqrt(len(paragraph_counts))
                },
                sample_size=len(df),
                confidence_interval=(
                    mean_paragraphs - 1.96 * std_paragraphs / np.sqrt(len(paragraph_counts)),
                    mean_paragraphs + 1.96 * std_paragraphs / np.sqrt(len(paragraph_counts))
                ),
                journals_analyzed=df['journal'].unique().tolist(),
                domains_analyzed=df['domain'].unique().tolist(),
                validation_score=0.8
            ))
        
        # Pattern 2: Conceptual breadth progression
        valid_slopes = df['breadth_progression_slope'].dropna()
        if len(valid_slopes) > 10:
            mean_slope = np.mean(valid_slopes)
            patterns.append(EmpiricalPattern(
                pattern_id="empirical_breadth_progression",
                pattern_type="structural",
                description=f"Conceptual breadth progression slope: {mean_slope:.3f}",
                statistical_evidence={
                    'mean_slope': mean_slope,
                    'negative_slope_ratio': np.mean(valid_slopes < 0)
                },
                sample_size=len(valid_slopes),
                confidence_interval=(
                    mean_slope - 1.96 * np.std(valid_slopes) / np.sqrt(len(valid_slopes)),
                    mean_slope + 1.96 * np.std(valid_slopes) / np.sqrt(len(valid_slopes))
                ),
                journals_analyzed=df['journal'].unique().tolist(),
                domains_analyzed=df['domain'].unique().tolist(),
                validation_score=0.75
            ))
        
        return patterns
    
    def _identify_argument_patterns(self, df: pd.DataFrame) -> List[EmpiricalPattern]:
        """Identify argumentation structure patterns."""
        patterns = []
        
        # Analyze argumentation structure distribution
        arg_counts = df['argumentation_structure'].value_counts()
        total = len(df)
        
        for arg_type, count in arg_counts.items():
            proportion = count / total
            if count >= 10:  # Minimum sample size
                patterns.append(EmpiricalPattern(
                    pattern_id=f"empirical_argument_{arg_type}",
                    pattern_type="argumentative",
                    description=f"{arg_type.replace('_', ' ').title()} structure used in {proportion:.1%} of papers",
                    statistical_evidence={
                        'proportion': proportion,
                        'count': count,
                        'confidence_95': 1.96 * np.sqrt(proportion * (1 - proportion) / total)
                    },
                    sample_size=count,
                    confidence_interval=(
                        max(0, proportion - 1.96 * np.sqrt(proportion * (1 - proportion) / total)),
                        min(1, proportion + 1.96 * np.sqrt(proportion * (1 - proportion) / total))
                    ),
                    journals_analyzed=df['journal'].unique().tolist(),
                    domains_analyzed=df['domain'].unique().tolist(),
                    validation_score=0.7
                ))
        
        return patterns
    
    def _identify_journal_patterns(self, df: pd.DataFrame) -> List[EmpiricalPattern]:
        """Identify journal-specific patterns if sufficient data exists."""
        patterns = []
        
        journal_counts = df['journal'].value_counts()
        
        for journal, count in journal_counts.items():
            if count >= 10:  # Minimum sample size for journal-specific analysis
                journal_data = df[df['journal'] == journal]
                
                # Analyze paragraph count for this journal
                mean_paragraphs = journal_data['total_paragraphs'].mean()
                
                patterns.append(EmpiricalPattern(
                    pattern_id=f"journal_{journal.lower().replace(' ', '_')}_structure",
                    pattern_type="journal_specific",
                    description=f"{journal}: Average {mean_paragraphs:.1f} paragraphs in introductions",
                    statistical_evidence={
                        'mean_paragraphs': mean_paragraphs,
                        'sample_size': count
                    },
                    sample_size=count,
                    confidence_interval=(0, 0),  # Would need more sophisticated calculation
                    journals_analyzed=[journal],
                    domains_analyzed=journal_data['domain'].unique().tolist(),
                    validation_score=0.6  # Lower confidence for journal-specific
                ))
        
        return patterns
    
    def _calculate_slope(self, values: List[float]) -> float:
        """Calculate the slope of a progression (positive = increasing, negative = decreasing)."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_journal_name(self, doc: ProcessedDocument) -> str:
        """Extract journal name from document metadata."""
        return doc.metadata.get('journal', 'Unknown')
    
    def _classify_research_domain(self, text: str) -> str:
        """Classify research domain based on content."""
        text_lower = text.lower()
        
        domain_keywords = {
            'neurosurgery': ['surgery', 'surgical', 'tumor', 'resection', 'operative'],
            'cognitive_neuroscience': ['cognitive', 'behavior', 'memory', 'attention'],
            'neuroimaging': ['fmri', 'mri', 'imaging', 'scan', 'voxel'],
            'cellular_neuroscience': ['neuron', 'synaptic', 'cellular', 'molecular'],
            'clinical_neuroscience': ['patient', 'clinical', 'disorder', 'treatment']
        }
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        else:
            return 'general_neuroscience'
    
    def _load_domain_indicators(self) -> Dict[str, List[str]]:
        """Load domain-specific terminology indicators."""
        return {
            'neuroscience_terms': [
                'brain', 'neural', 'neuron', 'synapse', 'cortex', 'hippocampus',
                'amygdala', 'cerebellum', 'neurotransmitter', 'dopamine', 'serotonin',
                'plasticity', 'connectivity', 'network', 'functional', 'structural',
                'cognitive', 'behavioral', 'motor', 'sensory', 'visual', 'auditory'
            ]
        }
    
    def _load_existing_patterns(self):
        """Load previously saved empirical patterns."""
        pattern_file = self.data_dir / "empirical_patterns.json"
        if pattern_file.exists():
            try:
                with open(pattern_file, 'r') as f:
                    data = json.load(f)
                    # Load patterns from saved data
                    logger.info(f"Loaded existing empirical patterns from {pattern_file}")
            except Exception as e:
                logger.warning(f"Failed to load existing patterns: {e}")
    
    def _save_collection_results(self, results: DataCollectionResult):
        """Save data collection results."""
        results_file = self.data_dir / f"collection_results_{results.collection_date.strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            # Convert to serializable format
            data = {
                'total_papers_analyzed': results.total_papers_analyzed,
                'successful_extractions': results.successful_extractions,
                'failed_extractions': results.failed_extractions,
                'journals_covered': results.journals_covered,
                'domains_covered': results.domains_covered,
                'patterns_count': len(results.patterns_identified),
                'collection_date': results.collection_date.isoformat()
            }
            
            with open(results_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved collection results to {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save collection results: {e}")
    
    def _detect_empirical_structural_patterns(self, results: List[AnalysisResult]) -> List[EmpiricalPattern]:
        """Detect structural patterns using empirical methods."""
        if len(results) < 10:
            logger.warning(f"Insufficient data for structural pattern detection: {len(results)} papers")
            return []
        
        patterns = []
        
        # Calculate paragraph count statistics
        paragraph_counts = [len(result.extracted_sections.get('introduction', '').split('\n\n')) 
                          for result in results if result.extracted_sections.get('introduction')]
        
        if paragraph_counts:
            mean_paragraphs = np.mean(paragraph_counts)
            std_paragraphs = np.std(paragraph_counts)
            
            # Create empirical pattern for paragraph count
            pattern = EmpiricalPattern(
                pattern_id=f"structural_paragraphs_{datetime.now().strftime('%Y%m%d')}",
                pattern_type="paragraph_count",
                description=f"Optimal paragraph count based on {len(paragraph_counts)} analyzed papers",
                statistical_evidence={
                    "mean": float(mean_paragraphs),
                    "std": float(std_paragraphs),
                    "min": float(np.min(paragraph_counts)),
                    "max": float(np.max(paragraph_counts)),
                    "median": float(np.median(paragraph_counts))
                },
                sample_size=len(paragraph_counts),
                confidence_interval=self._calculate_confidence_interval(paragraph_counts),
                journals_analyzed=list(set(r.journal for r in results if r.journal)),
                domains_analyzed=list(set(r.domain for r in results if r.domain)),
                validation_score=self._calculate_validation_score(paragraph_counts)
            )
            patterns.append(pattern)
        
        # Calculate sentence length patterns
        sentence_lengths = []
        for result in results:
            intro_text = result.extracted_sections.get('introduction', '')
            if intro_text:
                sentences = [s.strip() for s in intro_text.split('.') if s.strip()]
                sentence_lengths.extend([len(s.split()) for s in sentences])
        
        if sentence_lengths:
            mean_length = np.mean(sentence_lengths)
            std_length = np.std(sentence_lengths)
            
            pattern = EmpiricalPattern(
                pattern_id=f"structural_sentences_{datetime.now().strftime('%Y%m%d')}",
                pattern_type="sentence_length",
                description=f"Optimal sentence length based on {len(sentence_lengths)} analyzed sentences",
                statistical_evidence={
                    "mean": float(mean_length),
                    "std": float(std_length),
                    "recommended_range": [float(mean_length - std_length), float(mean_length + std_length)]
                },
                sample_size=len(sentence_lengths),
                confidence_interval=self._calculate_confidence_interval(sentence_lengths),
                journals_analyzed=list(set(r.journal for r in results if r.journal)),
                domains_analyzed=list(set(r.domain for r in results if r.domain)),
                validation_score=self._calculate_validation_score(sentence_lengths)
            )
            patterns.append(pattern)
        
        logger.info(f"Detected {len(patterns)} structural patterns from {len(results)} papers")
        return patterns
    
    def _detect_empirical_argumentative_patterns(self, results: List[AnalysisResult]) -> List[EmpiricalPattern]:
        """Detect argumentative patterns using empirical methods."""
        if len(results) < 10:
            return []
        
        patterns = []
        
        # Analyze problem-gap-solution structure frequency
        problem_gap_solution_count = 0
        hypothesis_driven_count = 0
        
        # Keywords for different argumentation patterns
        problem_keywords = ['problem', 'challenge', 'difficulty', 'issue']
        gap_keywords = ['gap', 'unknown', 'unclear', 'limited', 'lacking']
        solution_keywords = ['propose', 'investigate', 'examine', 'study']
        hypothesis_keywords = ['hypothesize', 'predict', 'expect', 'propose that']
        
        for result in results:
            intro_text = result.extracted_sections.get('introduction', '').lower()
            if not intro_text:
                continue
            
            # Check for problem-gap-solution pattern
            has_problem = any(keyword in intro_text for keyword in problem_keywords)
            has_gap = any(keyword in intro_text for keyword in gap_keywords)
            has_solution = any(keyword in intro_text for keyword in solution_keywords)
            
            if has_problem and has_gap and has_solution:
                problem_gap_solution_count += 1
            
            # Check for hypothesis-driven pattern
            has_hypothesis = any(keyword in intro_text for keyword in hypothesis_keywords)
            if has_hypothesis:
                hypothesis_driven_count += 1
        
        # Create pattern for argumentation structure frequency
        total_analyzed = len([r for r in results if r.extracted_sections.get('introduction')])
        if total_analyzed > 0:
            pgs_frequency = problem_gap_solution_count / total_analyzed
            hypothesis_frequency = hypothesis_driven_count / total_analyzed
            
            pattern = EmpiricalPattern(
                pattern_id=f"argumentative_structure_{datetime.now().strftime('%Y%m%d')}",
                pattern_type="argumentation_structure",
                description=f"Argumentation structure frequencies based on {total_analyzed} analyzed papers",
                statistical_evidence={
                    "problem_gap_solution_frequency": float(pgs_frequency),
                    "hypothesis_driven_frequency": float(hypothesis_frequency),
                    "problem_gap_solution_count": problem_gap_solution_count,
                    "hypothesis_driven_count": hypothesis_driven_count,
                    "total_analyzed": total_analyzed
                },
                sample_size=total_analyzed,
                confidence_interval=self._calculate_proportion_confidence_interval(pgs_frequency, total_analyzed),
                journals_analyzed=list(set(r.journal for r in results if r.journal)),
                domains_analyzed=list(set(r.domain for r in results if r.domain)),
                validation_score=0.8 if total_analyzed >= 50 else 0.6
            )
            patterns.append(pattern)
        
        logger.info(f"Detected {len(patterns)} argumentative patterns from {len(results)} papers")
        return patterns
    
    def _detect_empirical_transitional_patterns(self, results: List[AnalysisResult]) -> List[EmpiricalPattern]:
        """Detect transitional patterns using empirical methods."""
        if len(results) < 10:
            return []
        
        patterns = []
        
        # Common transition phrases
        transitions = {
            'addition': ['furthermore', 'moreover', 'additionally', 'also', 'in addition'],
            'contrast': ['however', 'nevertheless', 'in contrast', 'on the other hand', 'despite'],
            'causation': ['therefore', 'consequently', 'thus', 'as a result', 'hence'],
            'temporal': ['subsequently', 'previously', 'following', 'prior to', 'meanwhile']
        }
        
        transition_frequencies = {category: 0 for category in transitions}
        total_transitions = 0
        total_papers = 0
        
        for result in results:
            intro_text = result.extracted_sections.get('introduction', '').lower()
            if not intro_text:
                continue
            
            total_papers += 1
            paper_transitions = 0
            
            for category, phrases in transitions.items():
                for phrase in phrases:
                    count = intro_text.count(phrase)
                    transition_frequencies[category] += count
                    paper_transitions += count
                    total_transitions += count
        
        if total_papers > 0:
            # Calculate transition density (transitions per paper)
            avg_transitions_per_paper = total_transitions / total_papers
            
            # Calculate relative frequencies of each transition type
            relative_frequencies = {
                category: count / max(total_transitions, 1) 
                for category, count in transition_frequencies.items()
            }
            
            pattern = EmpiricalPattern(
                pattern_id=f"transitional_patterns_{datetime.now().strftime('%Y%m%d')}",
                pattern_type="transition_usage",
                description=f"Transition pattern usage based on {total_papers} analyzed papers",
                statistical_evidence={
                    "avg_transitions_per_paper": float(avg_transitions_per_paper),
                    "total_transitions_analyzed": total_transitions,
                    "transition_frequencies": {k: int(v) for k, v in transition_frequencies.items()},
                    "relative_frequencies": relative_frequencies
                },
                sample_size=total_papers,
                confidence_interval=(
                    float(max(0, avg_transitions_per_paper - 1)), 
                    float(avg_transitions_per_paper + 1)
                ),
                journals_analyzed=list(set(r.journal for r in results if r.journal)),
                domains_analyzed=list(set(r.domain for r in results if r.domain)),
                validation_score=0.8 if total_papers >= 30 else 0.6
            )
            patterns.append(pattern)
        
        logger.info(f"Detected {len(patterns)} transitional patterns from {len(results)} papers")
        return patterns
    
    def _calculate_confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for numerical data."""
        if len(data) < 2:
            return (0.0, 0.0)
        
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1)
        n = len(data)
        
        # t-distribution critical value (approximation for large n)
        t_critical = 1.96 if n > 30 else 2.0  # Conservative estimate
        
        margin_error = t_critical * (std_val / np.sqrt(n))
        
        return (float(mean_val - margin_error), float(mean_val + margin_error))
    
    def _calculate_proportion_confidence_interval(self, proportion: float, sample_size: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for proportion data."""
        if sample_size < 2:
            return (0.0, 1.0)
        
        # Wilson score interval (more accurate than normal approximation)
        z = 1.96  # 95% confidence
        n = sample_size
        p = proportion
        
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator
        
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
        
        return (float(lower), float(upper))
    
    def _calculate_validation_score(self, data: List[float]) -> float:
        """Calculate validation score based on data quality."""
        if len(data) < 10:
            return 0.3
        elif len(data) < 30:
            return 0.6
        elif len(data) < 50:
            return 0.75
        else:
            return 0.9