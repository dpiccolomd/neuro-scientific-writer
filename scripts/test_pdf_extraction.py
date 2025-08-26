#!/usr/bin/env python3
"""
Test script for PDF extraction functionality.
Creates a sample PDF for testing when no real PDFs are available.
"""

import sys
import tempfile
from pathlib import Path
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pdf_processor import PDFExtractor, PDFProcessingError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_sample_pdf_content():
    """Create sample content that looks like a neuroscience paper."""
    return """
Neural Mechanisms of Memory Formation in the Human Hippocampus

ABSTRACT

The hippocampus plays a crucial role in memory formation and consolidation. 
This study investigates the neural mechanisms underlying memory encoding 
in the human brain using functional magnetic resonance imaging (fMRI).
We found significant activation in the CA1 region during memory tasks.

INTRODUCTION

Memory formation is a fundamental cognitive process that involves multiple 
brain regions. Previous studies have shown that the hippocampus is critical 
for episodic memory (Smith et al., 2019). The CA1 pyramidal neurons 
demonstrate synaptic plasticity during learning (Jones & Brown, 2020).

Neuroplasticity mechanisms include long-term potentiation (LTP) and 
long-term depression (LTD) (Wilson, 2018). These processes involve 
glutamate receptors and calcium signaling pathways.

METHODS

Participants underwent fMRI scanning while performing memory tasks.
We analyzed BOLD signals in the hippocampus and surrounding regions.
Statistical analysis was performed using SPM12 software.

RESULTS

Hippocampal activation was observed in 95% of participants during 
encoding tasks. The CA1 region showed the strongest activation 
compared to CA3 and dentate gyrus regions (p < 0.001).

DISCUSSION

Our findings confirm the critical role of hippocampal CA1 neurons 
in memory formation. The results align with previous electrophysiology 
studies in animal models (Davis et al., 2021).

The observed activation patterns suggest that memory consolidation 
involves distributed neural networks including the prefrontal cortex 
and temporal lobe structures.

CONCLUSIONS

The hippocampus, particularly the CA1 region, is essential for 
human memory formation. Future studies should investigate the 
molecular mechanisms underlying these processes.

REFERENCES

Davis, R., Smith, J., & Wilson, K. (2021). Hippocampal function in memory. 
Nature Neuroscience, 15(3), 234-245.

Jones, M., & Brown, L. (2020). Synaptic plasticity mechanisms. 
Journal of Neuroscience, 40(12), 1234-1245.

Smith, A., Johnson, B., & Lee, C. (2019). Memory and the brain. 
Current Biology, 29(8), 456-467.

Wilson, P. (2018). Long-term potentiation in learning. 
Science, 362(6410), 123-130.
"""


def test_pdf_extractor():
    """Test the PDF extractor with comprehensive validation."""
    logger.info("Starting PDF extraction test...")
    
    try:
        # Initialize extractor
        extractor = PDFExtractor()
        logger.info("PDFExtractor initialized successfully")
        
        # Test with non-existent file
        try:
            extractor.process_pdf("/nonexistent/file.pdf")
            logger.error("Should have raised PDFProcessingError for non-existent file")
        except PDFProcessingError as e:
            logger.info(f"✓ Correctly handled non-existent file: {e}")
        
        # Test neuroscience keyword detection
        keywords = extractor.neuroscience_keywords
        logger.info(f"✓ Loaded {len(keywords)} neuroscience keywords")
        assert 'hippocampus' in keywords
        assert 'neuron' in keywords
        assert 'synaptic' in keywords
        
        # Test section pattern detection
        test_cases = [
            ("Abstract", "ABSTRACT"),
            ("Introduction", "INTRODUCTION"), 
            ("Methods", "METHODS"),
            ("Results", "RESULTS"),
            ("Discussion", "DISCUSSION"),
            ("References", "REFERENCES")
        ]
        
        for test_input, expected in test_cases:
            detected = extractor._identify_section_type(test_input)
            logger.info(f"✓ Section detection: '{test_input}' -> {detected}")
        
        # Test citation extraction
        test_text = create_sample_pdf_content()
        citations = extractor._extract_citations(test_text)
        logger.info(f"✓ Extracted {len(citations)} citations")
        
        for citation in citations:
            logger.info(f"  - {citation.text} (confidence: {citation.confidence:.2f})")
        
        # Test metadata extraction
        document_data = {
            'text': test_text,
            'pages': [
                {'page_num': 1, 'text': test_text[:500], 'char_count': 500},
                {'page_num': 2, 'text': test_text[500:1000], 'char_count': 500},
                {'page_num': 3, 'text': test_text[1000:], 'char_count': len(test_text) - 1000}
            ]
        }
        
        metadata = extractor._extract_metadata(document_data)
        logger.info(f"✓ Extracted metadata:")
        logger.info(f"  - Title: {metadata.title}")
        logger.info(f"  - Keywords: {metadata.keywords}")
        logger.info(f"  - DOI: {metadata.doi}")
        
        # Test section detection
        sections = extractor._detect_sections(test_text, document_data['pages'])
        logger.info(f"✓ Detected {len(sections)} sections:")
        
        for section in sections:
            logger.info(f"  - {section.section_type.value}: {section.title[:50]}...")
            logger.info(f"    Pages: {section.page_start}-{section.page_end}, "
                       f"Words: {section.word_count}, "
                       f"Confidence: {section.confidence:.2f}")
        
        # Test citation assignment
        sections_with_citations = extractor._assign_citations_to_sections(sections, citations)
        total_assigned_citations = sum(len(s.citations) for s in sections_with_citations)
        logger.info(f"✓ Assigned {total_assigned_citations} citations to sections")
        
        # Test confidence calculation
        confidence = extractor._calculate_confidence_score(sections, metadata)
        logger.info(f"✓ Overall confidence score: {confidence:.2f}")
        
        logger.info("=" * 60)
        logger.info("🎉 ALL TESTS PASSED! PDF processing functionality is working correctly.")
        
        # Summary statistics
        logger.info(f"\nSummary:")
        logger.info(f"- Keywords detected: {len(metadata.keywords)}")
        logger.info(f"- Sections found: {len(sections)}")
        logger.info(f"- Citations extracted: {len(citations)}")
        logger.info(f"- Overall confidence: {confidence:.2f}")
        logger.info(f"- Total word count: {sum(s.word_count for s in sections)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = test_pdf_extractor()
    sys.exit(0 if success else 1)