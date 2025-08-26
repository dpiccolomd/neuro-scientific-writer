#!/usr/bin/env python3
"""
Test script for NLP text analysis functionality.
Tests the NeuroTextAnalyzer with sample neuroscience text.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis import NeuroTextAnalyzer, WritingPatternDetector, AnalysisError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_sample_neuroscience_texts():
    """Create sample neuroscience texts for testing."""
    return {
        "introduction_sample": """
        The human brain is one of the most complex structures in the known universe, containing approximately 86 billion neurons interconnected through trillions of synapses. Understanding the neural mechanisms underlying cognitive functions has been a central focus of neuroscience research for decades. However, despite significant advances in neuroimaging techniques such as functional magnetic resonance imaging (fMRI) and electroencephalography (EEG), the precise mechanisms by which neural networks support complex behaviors remain poorly understood.
        
        Memory formation, in particular, has emerged as a critical area of investigation. Previous studies have shown that the hippocampus plays a crucial role in episodic memory consolidation (Smith et al., 2019; Johnson & Brown, 2020). The CA1 region of the hippocampus demonstrates remarkable synaptic plasticity, with long-term potentiation (LTP) serving as a proposed cellular mechanism for memory storage (Davis et al., 2021). Furthermore, recent evidence suggests that neurogenesis in the dentate gyrus may contribute to pattern separation and memory discrimination (Wilson & Lee, 2018).
        
        We hypothesize that hippocampal-cortical interactions during sleep facilitate memory consolidation through synchronized oscillatory activity. The present study aims to investigate the neural mechanisms underlying memory consolidation using simultaneous fMRI and EEG recordings during different sleep stages.
        """,
        
        "methods_sample": """
        Twenty-four healthy participants (12 female, mean age 24.3 ± 3.2 years) were recruited for this study. All participants provided informed consent and were screened for neurological and psychiatric conditions. Inclusion criteria included normal hearing, right-handedness, and no history of sleep disorders.
        
        Participants underwent simultaneous EEG-fMRI recording during overnight sleep sessions. EEG data were acquired using a 32-channel MR-compatible system with a sampling rate of 5000 Hz. Functional MRI data were collected using a 3T scanner with a gradient-echo EPI sequence (TR = 2000 ms, TE = 30 ms, voxel size = 3×3×3 mm³).
        
        Sleep stages were scored according to standard criteria using 30-second epochs. Statistical analysis was performed using SPM12 for fMRI data and EEGLAB for electrophysiological data. Hippocampal regions of interest were defined anatomically using the Harvard-Oxford atlas.
        """,
        
        "results_sample": """
        Sleep stage classification revealed typical sleep architecture with participants spending 23.4 ± 4.1% of time in REM sleep and 18.7 ± 3.8% in slow-wave sleep. Hippocampal activation during slow-wave sleep showed significant correlations with memory performance (r = 0.67, p < 0.001).
        
        Analysis of hippocampal-cortical connectivity revealed increased functional connectivity between the hippocampus and prefrontal cortex during slow-wave sleep compared to REM sleep (t(23) = 4.23, p < 0.001, Cohen's d = 0.86). This increased connectivity was positively correlated with overnight memory retention (r = 0.58, p < 0.01).
        
        EEG analysis showed synchronized theta oscillations (4-8 Hz) in hippocampal and cortical regions during memory consolidation periods. The results demonstrate that hippocampal-cortical synchronization facilitates memory consolidation through coordinated oscillatory activity.
        """,
        
        "discussion_sample": """
        These findings provide compelling evidence for the role of hippocampal-cortical interactions in memory consolidation. Our results are consistent with previous research demonstrating the importance of slow-wave sleep for memory processing (Anderson et al., 2020). The observed increase in functional connectivity between hippocampus and prefrontal cortex suggests that these regions work in concert to facilitate memory stabilization.
        
        The synchronized theta oscillations observed during memory consolidation periods may represent a neural mechanism by which information is transferred from hippocampal to cortical storage sites. This interpretation aligns with systems consolidation theory, which proposes that memories gradually become less dependent on the hippocampus over time (Miller & Thompson, 2019).
        
        However, several limitations should be considered. The sample size was relatively small, and future studies should include larger cohorts to increase statistical power. Additionally, we cannot rule out the possibility that other brain regions contribute to the observed connectivity patterns. Future research should investigate the temporal dynamics of memory consolidation using higher-resolution imaging techniques.
        
        These findings have important implications for understanding memory disorders such as Alzheimer's disease, where hippocampal dysfunction may disrupt normal consolidation processes. The results may also inform therapeutic approaches for enhancing memory function in clinical populations.
        """
    }


def test_text_analyzer():
    """Test the NeuroTextAnalyzer comprehensively."""
    logger.info("Starting text analysis tests...")
    
    try:
        # Initialize analyzer
        analyzer = NeuroTextAnalyzer()
        logger.info("NeuroTextAnalyzer initialized successfully")
        
        # Test with different text samples
        sample_texts = create_sample_neuroscience_texts()
        results = []
        
        for text_type, text_content in sample_texts.items():
            logger.info(f"Analyzing {text_type}...")
            
            try:
                result = analyzer.analyze_text(
                    text=text_content,
                    text_id=f"sample_{text_type}",
                    source_type="section"
                )
                results.append(result)
                
                # Log analysis results
                logger.info(f"✓ Analysis completed for {text_type}:")
                logger.info(f"  - Sentences: {result.total_sentences}")
                logger.info(f"  - Words: {result.total_words}")
                logger.info(f"  - Neuro terms: {len(result.neuro_terms)}")
                logger.info(f"  - Writing patterns: {len(result.writing_patterns)}")
                logger.info(f"  - Quality score: {result.overall_quality_score:.3f}")
                logger.info(f"  - Processing time: {result.processing_time:.3f}s")
                
                # Test specific aspects
                logger.info(f"  - Style characteristics:")
                logger.info(f"    • Formality: {result.style_characteristics.formality_level:.3f}")
                logger.info(f"    • Technical density: {result.style_characteristics.technical_density:.3f}")
                logger.info(f"    • Citation density: {result.style_characteristics.citation_density:.3f}")
                logger.info(f"    • Writing style: {result.style_characteristics.writing_style.value}")
                
                logger.info(f"  - Terminology analysis:")
                logger.info(f"    • Total terms: {result.terminology_analysis.total_terms}")
                logger.info(f"    • Unique terms: {result.terminology_analysis.unique_terms}")
                logger.info(f"    • Complexity: {result.terminology_analysis.complexity_level}")
                logger.info(f"    • Categories: {list(result.terminology_analysis.term_categories.keys())}")
                
                logger.info(f"  - Coherence metrics:")
                logger.info(f"    • Overall coherence: {result.coherence_metrics.overall_coherence:.3f}")
                logger.info(f"    • Lexical cohesion: {result.coherence_metrics.lexical_cohesion:.3f}")
                logger.info(f"    • Transition quality: {result.coherence_metrics.transition_quality:.3f}")
                
                # Show sample neuroscience terms found
                if result.neuro_terms:
                    logger.info(f"  - Sample neuro terms:")
                    for term in result.neuro_terms[:5]:  # Show first 5
                        logger.info(f"    • {term.term} ({term.category}) - confidence: {term.confidence:.3f}")
                
                logger.info("")
                
            except Exception as e:
                logger.error(f"Failed to analyze {text_type}: {e}")
                return False
        
        # Test pattern detector
        logger.info("Testing WritingPatternDetector...")
        pattern_detector = WritingPatternDetector()
        
        detected_patterns = pattern_detector.detect_patterns(results)
        logger.info(f"✓ Detected {len(detected_patterns)} writing patterns:")
        
        for pattern in detected_patterns:
            logger.info(f"  - {pattern.pattern_type}: {pattern.description}")
            logger.info(f"    Frequency: {pattern.frequency:.3f}, Confidence: {pattern.confidence:.3f}")
            logger.info(f"    Sections: {pattern.section_types}")
        
        # Test template generation
        template = pattern_detector.generate_template_from_patterns(detected_patterns)
        logger.info(f"✓ Generated writing template:")
        logger.info(f"  - Total patterns: {template['pattern_summary']['total_patterns']}")
        logger.info(f"  - Primary focus: {template['pattern_summary']['primary_focus']}")
        logger.info(f"  - Writing complexity: {template['pattern_summary']['writing_complexity']}")
        
        # Test error handling
        logger.info("Testing error handling...")
        
        try:
            analyzer.analyze_text("", "empty_test")
            logger.error("Should have raised AnalysisError for empty text")
            return False
        except AnalysisError:
            logger.info("✓ Correctly handled empty text error")
        
        # Test data serialization
        logger.info("Testing data serialization...")
        for result in results:
            result_dict = result.to_dict()
            assert isinstance(result_dict, dict)
            assert "text_id" in result_dict
            assert "sentences" in result_dict
            assert "neuro_terms" in result_dict
        logger.info("✓ Data serialization working correctly")
        
        # Summary statistics
        logger.info("=" * 60)
        logger.info("🎉 ALL TEXT ANALYSIS TESTS PASSED!")
        logger.info(f"\nSummary Statistics:")
        
        total_sentences = sum(r.total_sentences for r in results)
        total_words = sum(r.total_words for r in results)
        total_terms = sum(len(r.neuro_terms) for r in results)
        avg_quality = sum(r.overall_quality_score for r in results) / len(results)
        
        logger.info(f"- Total texts analyzed: {len(results)}")
        logger.info(f"- Total sentences: {total_sentences}")
        logger.info(f"- Total words: {total_words}")
        logger.info(f"- Total neuro terms: {total_terms}")
        logger.info(f"- Average quality score: {avg_quality:.3f}")
        logger.info(f"- Patterns detected: {len(detected_patterns)}")
        
        # Test performance
        total_processing_time = sum(r.processing_time for r in results)
        words_per_second = total_words / total_processing_time if total_processing_time > 0 else 0
        logger.info(f"- Total processing time: {total_processing_time:.3f}s")
        logger.info(f"- Processing speed: {words_per_second:.0f} words/second")
        
        return True
        
    except Exception as e:
        logger.error(f"Text analysis test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = test_text_analyzer()
    sys.exit(0 if success else 1)