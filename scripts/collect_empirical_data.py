#!/usr/bin/env python3
"""
Empirical Data Collection Script

This script collects empirical data from published neuroscience papers to build
statistically validated writing patterns. It replaces naive rule-based assumptions
with actual analysis of successful publications.

Usage:
    python scripts/collect_empirical_data.py --input data/training_papers/ --min_papers 50
    
Requirements:
    - At least 50 papers for statistical validity
    - Papers should be from peer-reviewed neuroscience journals
    - PDFs should be high-quality with extractable text
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pdf_processor import PDFExtractor
from analysis.empirical_pattern_detector import EmpiricalPatternDetector, DataCollectionResult
from pdf_processor.models import ProcessedDocument

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_empirical_data(
    papers_directory: str,
    output_directory: str = "data/empirical_patterns",
    min_papers: int = 50,
    max_papers: int = 200
) -> DataCollectionResult:
    """
    Collect empirical data from a directory of research papers.
    
    Args:
        papers_directory: Directory containing PDF papers
        output_directory: Directory to save results
        min_papers: Minimum number of papers for validity
        max_papers: Maximum papers to process (for performance)
        
    Returns:
        DataCollectionResult with analysis and patterns
    """
    logger.info("Starting empirical data collection for pattern detection")
    logger.info("=" * 60)
    
    # Initialize components
    pdf_extractor = PDFExtractor()
    pattern_detector = EmpiricalPatternDetector(output_directory)
    
    # Find PDF files
    papers_path = Path(papers_directory)
    if not papers_path.exists():
        raise ValueError(f"Papers directory does not exist: {papers_directory}")
    
    pdf_files = list(papers_path.glob("*.pdf"))
    
    if len(pdf_files) < min_papers:
        raise ValueError(
            f"Insufficient papers for empirical analysis: {len(pdf_files)} < {min_papers}\n"
            f"Need at least {min_papers} peer-reviewed neuroscience papers for statistical validity."
        )
    
    if len(pdf_files) > max_papers:
        logger.warning(f"Found {len(pdf_files)} papers, limiting to {max_papers} for performance")
        pdf_files = pdf_files[:max_papers]
    
    logger.info(f"Found {len(pdf_files)} PDF files for analysis")
    
    # Process papers
    processed_documents = []
    successful = 0
    failed = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        logger.info(f"Processing {i}/{len(pdf_files)}: {pdf_file.name}")
        
        try:
            # Extract document
            doc = pdf_extractor.process_pdf(str(pdf_file))
            
            if doc and doc.introduction_section:
                processed_documents.append(doc)
                successful += 1
                logger.info(f"  ✓ Successfully extracted introduction ({len(doc.introduction_section.content)} chars)")
            else:
                failed += 1
                logger.warning(f"  ✗ No introduction section found")
                
        except Exception as e:
            failed += 1
            logger.error(f"  ✗ Processing failed: {e}")
    
    logger.info(f"\nProcessing complete: {successful} successful, {failed} failed")
    
    if successful < min_papers:
        raise ValueError(
            f"Insufficient successful extractions: {successful} < {min_papers}\n"
            "Need higher quality PDFs with extractable introduction sections."
        )
    
    # Collect empirical data and identify patterns
    logger.info("\nStarting empirical pattern analysis...")
    logger.info("=" * 60)
    
    collection_result = pattern_detector.collect_empirical_data(
        processed_documents, 
        min_sample_size=min_papers
    )
    
    # Print summary
    print_collection_summary(collection_result)
    
    return collection_result


def print_collection_summary(result: DataCollectionResult):
    """Print a summary of the data collection results."""
    print("\n" + "=" * 60)
    print("EMPIRICAL DATA COLLECTION SUMMARY")
    print("=" * 60)
    
    print(f"Papers Analyzed:       {result.total_papers_analyzed}")
    print(f"Successful Extractions: {result.successful_extractions}")
    print(f"Failed Extractions:     {result.failed_extractions}")
    print(f"Success Rate:          {result.successful_extractions/result.total_papers_analyzed:.1%}")
    
    print(f"\nJournals Covered ({len(result.journals_covered)}):")
    for journal in sorted(result.journals_covered)[:10]:  # Show first 10
        print(f"  - {journal}")
    if len(result.journals_covered) > 10:
        print(f"  ... and {len(result.journals_covered) - 10} more")
    
    print(f"\nResearch Domains ({len(result.domains_covered)}):")
    for domain in sorted(result.domains_covered):
        print(f"  - {domain.replace('_', ' ').title()}")
    
    print(f"\nEmpirical Patterns Identified: {len(result.patterns_identified)}")
    
    # Show statistically significant patterns
    significant_patterns = [p for p in result.patterns_identified if p.is_statistically_significant]
    print(f"Statistically Significant:    {len(significant_patterns)}")
    
    if significant_patterns:
        print("\nSignificant Patterns Found:")
        for pattern in significant_patterns[:5]:  # Show first 5
            print(f"  - {pattern.description}")
            print(f"    Sample size: {pattern.sample_size}, Validation: {pattern.validation_score:.2f}")
    
    # Structural analysis summary
    if result.structural_metrics:
        paragraphs = [m.total_paragraphs for m in result.structural_metrics]
        sentences = [m.total_sentences for m in result.structural_metrics]
        
        print(f"\nStructural Analysis:")
        print(f"  Average paragraphs per intro: {sum(paragraphs)/len(paragraphs):.1f}")
        print(f"  Average sentences per intro:  {sum(sentences)/len(sentences):.1f}")
        
        # Argumentation structure distribution
        arg_structures = [m.argumentation_structure for m in result.structural_metrics]
        from collections import Counter
        arg_counts = Counter(arg_structures)
        
        print(f"\nArgumentation Structures:")
        for arg_type, count in arg_counts.most_common():
            percentage = count / len(arg_structures) * 100
            print(f"  - {arg_type.replace('_', ' ').title()}: {count} papers ({percentage:.1f}%)")
    
    print("\n" + "=" * 60)
    print("EMPIRICAL FOUNDATION ESTABLISHED")
    print("Pattern detection now based on statistical analysis")
    print("of actual published neuroscience literature.")
    print("=" * 60)


def validate_paper_collection(papers_directory: str) -> bool:
    """Validate that the paper collection meets requirements for empirical analysis."""
    papers_path = Path(papers_directory)
    
    if not papers_path.exists():
        print(f"❌ Papers directory does not exist: {papers_directory}")
        return False
    
    pdf_files = list(papers_path.glob("*.pdf"))
    
    if len(pdf_files) < 50:
        print(f"❌ Insufficient papers: {len(pdf_files)} < 50")
        print("   Need at least 50 peer-reviewed neuroscience papers")
        print("   for statistically valid pattern detection.")
        return False
    
    print(f"✅ Found {len(pdf_files)} PDF files")
    print("✅ Sufficient sample size for empirical analysis")
    
    # Check file sizes (rough quality indicator)
    small_files = [f for f in pdf_files if f.stat().st_size < 100_000]  # < 100KB
    if len(small_files) > len(pdf_files) * 0.2:  # > 20% small files
        print(f"⚠️  Warning: {len(small_files)} files are very small (< 100KB)")
        print("   This may indicate low-quality or incomplete PDFs")
    
    return True


def main():
    """Main entry point for empirical data collection."""
    parser = argparse.ArgumentParser(
        description="Collect empirical data from neuroscience papers for pattern detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic collection with minimum 50 papers
  python scripts/collect_empirical_data.py --input data/training_papers/
  
  # High-quality analysis with 100+ papers
  python scripts/collect_empirical_data.py --input data/training_papers/ --min_papers 100
  
  # Validate collection before processing
  python scripts/collect_empirical_data.py --input data/training_papers/ --validate_only

Requirements for Statistical Validity:
  - Minimum 50 papers from peer-reviewed neuroscience journals
  - Papers should span multiple journals and research domains
  - PDFs should be high-quality with extractable text
  - Prefer recent papers (last 10 years) for current patterns
        """
    )
    
    parser.add_argument(
        '--input', 
        required=True,
        help='Directory containing PDF papers for analysis'
    )
    
    parser.add_argument(
        '--output',
        default='data/empirical_patterns',
        help='Output directory for empirical patterns (default: data/empirical_patterns)'
    )
    
    parser.add_argument(
        '--min_papers',
        type=int,
        default=50,
        help='Minimum number of papers required for analysis (default: 50)'
    )
    
    parser.add_argument(
        '--max_papers',
        type=int,
        default=200,
        help='Maximum number of papers to process (default: 200)'
    )
    
    parser.add_argument(
        '--validate_only',
        action='store_true',
        help='Only validate the paper collection, do not process'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Validate paper collection
        if not validate_paper_collection(args.input):
            sys.exit(1)
        
        if args.validate_only:
            print("✅ Paper collection validation passed")
            return
        
        # Run empirical data collection
        result = collect_empirical_data(
            papers_directory=args.input,
            output_directory=args.output,
            min_papers=args.min_papers,
            max_papers=args.max_papers
        )
        
        print(f"\n✅ Empirical data collection completed successfully")
        print(f"📊 {len(result.patterns_identified)} patterns identified")
        print(f"💾 Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()