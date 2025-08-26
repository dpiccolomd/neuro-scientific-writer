#!/usr/bin/env python3
"""
Test script for template generation functionality.
Tests the TemplateGenerator with sample analysis results and patterns.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis import NeuroTextAnalyzer, WritingPatternDetector
from template_engine import TemplateGenerator
from template_engine.models import TemplateType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_sample_analysis_data():
    """Create sample analysis results for testing."""
    sample_texts = {
        "introduction": """
        The human brain contains approximately 86 billion neurons interconnected through trillions of synapses, making it one of the most complex biological systems. Understanding neural mechanisms underlying cognitive functions has been a central focus of neuroscience research. However, despite advances in neuroimaging techniques such as fMRI and EEG, the precise mechanisms remain poorly understood.
        
        Previous studies have shown that the hippocampus plays a crucial role in memory formation (Smith et al., 2019). The CA1 region demonstrates synaptic plasticity through long-term potentiation (Davis et al., 2021). We hypothesize that hippocampal-cortical interactions during sleep facilitate memory consolidation. The present study aims to investigate these mechanisms using simultaneous EEG-fMRI recordings.
        """,
        
        "methods": """
        Twenty-four participants underwent EEG-fMRI recording during sleep. EEG data were acquired using a 32-channel system. Statistical analysis was performed using SPM12. Hippocampal regions were defined using anatomical atlases.
        """,
        
        "results": """
        Sleep analysis revealed typical architecture with 23% REM sleep. Hippocampal activation correlated with memory performance (r = 0.67, p < 0.001). Functional connectivity between hippocampus and prefrontal cortex increased during slow-wave sleep (t(23) = 4.23, p < 0.001).
        """,
        
        "discussion": """
        These findings demonstrate hippocampal-cortical interactions in memory consolidation. Results are consistent with previous research (Anderson et al., 2020). The synchronized oscillations may represent information transfer mechanisms. However, limitations include small sample size. Future research should investigate temporal dynamics.
        """
    }
    
    return sample_texts


def test_template_generator():
    """Test the template generation system comprehensively."""
    logger.info("Starting template generation tests...")
    
    try:
        # Initialize components
        analyzer = NeuroTextAnalyzer()
        pattern_detector = WritingPatternDetector()
        template_generator = TemplateGenerator()
        
        logger.info("✓ All components initialized successfully")
        
        # Create sample analysis data
        sample_texts = create_sample_analysis_data()
        
        # Analyze all sample texts
        analysis_results = []
        for section_name, text_content in sample_texts.items():
            result = analyzer.analyze_text(
                text=text_content,
                text_id=f"sample_{section_name}",
                source_type="section"
            )
            analysis_results.append(result)
            logger.info(f"✓ Analyzed {section_name}: {result.total_sentences} sentences, {result.total_words} words")
        
        # Detect writing patterns
        patterns = pattern_detector.detect_patterns(analysis_results)
        logger.info(f"✓ Detected {len(patterns)} writing patterns:")
        
        for pattern in patterns:
            logger.info(f"  - {pattern.pattern_type}: {pattern.description[:60]}...")
        
        # Test different template types
        template_types_to_test = [
            None,  # Auto-detect
            TemplateType.INTRODUCTION_FUNNEL,
            TemplateType.HYPOTHESIS_DRIVEN,
            TemplateType.METHODOLOGY_FOCUSED
        ]
        
        generated_templates = []
        
        for template_type in template_types_to_test:
            try:
                logger.info(f"Generating template: {template_type.value if template_type else 'AUTO-DETECT'}")
                
                template = template_generator.generate_template(
                    analysis_results=analysis_results,
                    detected_patterns=patterns,
                    template_type=template_type
                )
                
                generated_templates.append(template)
                
                logger.info(f"✓ Generated template: {template.template_id}")
                logger.info(f"  - Type: {template.metadata.template_type.value}")
                logger.info(f"  - Sections: {len(template.sections)}")
                logger.info(f"  - Word count: {template.total_word_count}")
                logger.info(f"  - Quality score: {template.overall_quality_score:.3f}")
                logger.info(f"  - Confidence: {template.metadata.confidence_score:.3f}")
                logger.info(f"  - Complexity: {template.metadata.complexity_level}")
                logger.info(f"  - Pattern coverage: {template.metadata.pattern_coverage:.3f}")
                
                # Test section details
                for i, section in enumerate(template.sections):
                    logger.info(f"  - Section {i+1}: {section.section_template.title}")
                    logger.info(f"    • Paragraphs: {len(section.section_template.paragraphs)}")
                    logger.info(f"    • Estimated words: {section.section_template.estimated_word_count}")
                    logger.info(f"    • Unfilled variables: {len(section.unfilled_variables)}")
                
                # Test paragraph details
                first_section = template.sections[0]
                logger.info(f"  - Paragraph types in first section:")
                for para in first_section.section_template.paragraphs:
                    logger.info(f"    • {para.paragraph_type.value}: {para.title}")
                    logger.info(f"      - Variables: {len(para.variables)}")
                    logger.info(f"      - Word range: {para.min_length}-{para.max_length}")
                    logger.info(f"      - Citation density: {para.citation_density}")
                
                logger.info("")
                
            except Exception as e:
                logger.error(f"Failed to generate template type {template_type}: {e}")
                return False
        
        # Test template serialization
        logger.info("Testing template serialization...")
        for template in generated_templates:
            template_dict = template.to_dict()
            assert isinstance(template_dict, dict)
            assert "template_id" in template_dict
            assert "metadata" in template_dict
            assert "sections" in template_dict
            assert "total_word_count" in template_dict
        
        logger.info("✓ Template serialization working correctly")
        
        # Test template content preview
        logger.info("Testing template content preview...")
        first_template = generated_templates[0]
        preview_content = first_template.sections[0].rendered_content
        
        assert len(preview_content) > 100  # Should have substantial content
        assert "Paragraph 1" in preview_content
        assert "Variables to fill" in preview_content
        assert "Word count" in preview_content
        
        logger.info("✓ Template content preview working correctly")
        
        # Test variable extraction
        logger.info("Testing variable extraction...")
        all_unfilled = first_template.get_unfilled_variables()
        logger.info(f"✓ Found {len(all_unfilled)} unfilled variables across template")
        
        # Show sample variables
        if all_unfilled:
            logger.info("Sample unfilled variables:")
            for var in all_unfilled[:5]:
                logger.info(f"  - {var}")
        
        # Test style guidelines
        logger.info("Testing style guidelines...")
        style_guidelines = first_template.style_guidelines
        logger.info(f"✓ Generated {len(style_guidelines)} style guidelines:")
        for key, value in style_guidelines.items():
            logger.info(f"  - {key}: {value}")
        
        # Test quality metrics
        logger.info("Testing quality metrics...")
        quality_metrics = first_template.quality_metrics
        logger.info(f"✓ Quality metrics:")
        for metric, score in quality_metrics.items():
            logger.info(f"  - {metric}: {score:.3f}")
        
        # Performance test
        logger.info("Testing performance...")
        import time
        start_time = time.time()
        
        for i in range(5):
            template = template_generator.generate_template(
                analysis_results=analysis_results,
                detected_patterns=patterns
            )
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 5
        logger.info(f"✓ Average generation time: {avg_time:.3f} seconds")
        
        # Summary
        logger.info("=" * 60)
        logger.info("🎉 ALL TEMPLATE GENERATION TESTS PASSED!")
        logger.info(f"\nSummary:")
        logger.info(f"- Templates generated: {len(generated_templates)}")
        logger.info(f"- Average word count: {sum(t.total_word_count for t in generated_templates) / len(generated_templates):.0f}")
        logger.info(f"- Average quality score: {sum(t.overall_quality_score for t in generated_templates) / len(generated_templates):.3f}")
        logger.info(f"- Average generation time: {avg_time:.3f}s")
        
        # Show a sample template preview
        logger.info(f"\n--- SAMPLE TEMPLATE PREVIEW ---")
        sample_preview = generated_templates[0].sections[0].rendered_content[:500]
        logger.info(sample_preview + "...")
        
        return True
        
    except Exception as e:
        logger.error(f"Template generation test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = test_template_generator()
    sys.exit(0 if success else 1)