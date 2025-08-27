#!/usr/bin/env python3
"""
Introduction Analysis Tool

Analyzes introduction text using empirically-trained patterns derived from published
neuroscience literature. Provides structural, argumentative, and quality assessments
based on statistical analysis of successful publications.

This tool uses REAL empirical data, not assumptions or simulated patterns.

Usage:
    python scripts/analyze_introduction.py --text "intro.txt" --patterns data/empirical_patterns/
    python scripts/analyze_introduction.py --text "intro.txt" --report analysis_report.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from analysis import NeuroTextAnalyzer, EmpiricalPatternDetector
from analysis.models import AnalysisResult
from quality_control import QualityValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_introduction(
    text_input: str,
    patterns_directory: str,
    output_file: Optional[str] = None,
    verbose: bool = False
) -> Dict:
    """
    Analyze introduction text using empirically-trained patterns.
    
    Args:
        text_input: Path to text file or text content
        patterns_directory: Directory containing trained empirical patterns
        output_file: Optional file to save analysis report
        verbose: Enable detailed output
        
    Returns:
        Dictionary containing analysis results
    """
    logger.info("Starting introduction analysis with empirical patterns")
    
    # Load text content
    if Path(text_input).exists():
        with open(text_input, 'r', encoding='utf-8') as f:
            text_content = f.read()
        source_file = text_input
    else:
        text_content = text_input
        source_file = "direct_input"
    
    if not text_content.strip():
        raise ValueError("No text content provided for analysis")
    
    # Initialize analyzers
    text_analyzer = NeuroTextAnalyzer()
    empirical_detector = EmpiricalPatternDetector(patterns_directory)
    quality_validator = QualityValidator()
    
    # Perform text analysis
    logger.info("Analyzing text structure and content")
    analysis_result = text_analyzer.analyze_text(
        text_content, 
        document_id="input_text",
        section_type="introduction"
    )
    
    # Detect empirical patterns (only if trained patterns exist)
    logger.info("Detecting empirical patterns")
    try:
        empirical_patterns = empirical_detector.detect_patterns_empirical(
            [analysis_result], 
            require_statistical_significance=True
        )
    except Exception as e:
        logger.warning(f"Empirical pattern detection failed: {e}")
        empirical_patterns = []
    
    # Quality validation
    logger.info("Performing quality validation")
    quality_report = quality_validator.validate_structure(text_content)
    
    # Compile comprehensive analysis report
    report = {
        "analysis_metadata": {
            "source": source_file,
            "analyzed_at": analysis_result.created_at.isoformat(),
            "text_length": len(text_content),
            "patterns_used": patterns_directory,
            "empirical_patterns_available": len(empirical_patterns) > 0
        },
        "structural_analysis": {
            "total_sentences": analysis_result.total_sentences,
            "total_words": analysis_result.word_count,
            "average_sentence_length": analysis_result.avg_sentence_length,
            "paragraph_count": len(text_content.split('\n\n')),
            "readability_score": analysis_result.readability_score
        },
        "content_analysis": {
            "neuroscience_terms": list(analysis_result.neuro_terms),
            "neuroscience_density": len(analysis_result.neuro_terms) / analysis_result.total_sentences,
            "key_concepts": analysis_result.key_terms[:10],
            "sentence_types": dict(analysis_result.sentence_types)
        },
        "empirical_pattern_analysis": {
            "patterns_detected": len(empirical_patterns),
            "statistically_significant": len([p for p in empirical_patterns if p.is_statistically_significant]),
            "pattern_details": [
                {
                    "pattern_id": p.pattern_id,
                    "pattern_type": p.pattern_type,
                    "description": p.description,
                    "validation_score": p.validation_score,
                    "sample_size": p.sample_size,
                    "confidence_interval": p.confidence_interval,
                    "journals_analyzed": p.journals_analyzed
                }
                for p in empirical_patterns
            ]
        },
        "quality_assessment": {
            "overall_score": quality_report.overall_score if hasattr(quality_report, 'overall_score') else 0.0,
            "structural_quality": quality_report.structure_score if hasattr(quality_report, 'structure_score') else 0.0,
            "recommendations": []
        }
    }
    
    # Add empirical recommendations if patterns available
    if empirical_patterns:
        report["empirical_recommendations"] = generate_empirical_recommendations(
            report, empirical_patterns
        )
    else:
        report["empirical_recommendations"] = [
            "No empirical patterns available. Train patterns using:",
            "python scripts/collect_empirical_data.py --input data/training_papers/",
            "or use Zotero integration:",
            "python scripts/train_from_zotero.py --collection 'Training Papers'"
        ]
    
    # Save report if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Analysis report saved to {output_file}")
    
    # Print summary
    print_analysis_summary(report, verbose)
    
    return report


def generate_empirical_recommendations(
    analysis_report: Dict, 
    empirical_patterns: List
) -> List[str]:
    """Generate recommendations based on empirical pattern analysis."""
    recommendations = []
    
    # Structural recommendations
    paragraph_count = analysis_report["structural_analysis"]["paragraph_count"]
    
    # Find structural patterns
    structural_patterns = [p for p in empirical_patterns if p.pattern_type == "structural"]
    for pattern in structural_patterns:
        if "paragraph" in pattern.pattern_id.lower():
            if "mean" in pattern.statistical_evidence:
                optimal_paragraphs = pattern.statistical_evidence["mean"]
                if abs(paragraph_count - optimal_paragraphs) > 1:
                    recommendations.append(
                        f"Consider adjusting paragraph count: current {paragraph_count}, "
                        f"empirical optimum {optimal_paragraphs:.1f} ± "
                        f"{pattern.statistical_evidence.get('std', 0):.1f} "
                        f"(based on {pattern.sample_size} papers)"
                    )
    
    # Argumentation recommendations
    arg_patterns = [p for p in empirical_patterns if p.pattern_type == "argumentative"]
    if arg_patterns:
        most_common_arg = max(arg_patterns, key=lambda p: p.statistical_evidence.get('proportion', 0))
        recommendations.append(
            f"Most successful argumentation structure in literature: "
            f"{most_common_arg.description} "
            f"(used in {most_common_arg.statistical_evidence.get('proportion', 0):.1%} of papers)"
        )
    
    # Content density recommendations
    neuro_density = analysis_report["content_analysis"]["neuroscience_density"]
    recommendations.append(
        f"Current neuroscience term density: {neuro_density:.2f} terms/sentence. "
        f"Compare against empirical benchmarks for your target journal."
    )
    
    return recommendations


def print_analysis_summary(report: Dict, verbose: bool = False):
    """Print a formatted summary of the analysis results."""
    print("\n" + "=" * 80)
    print("INTRODUCTION ANALYSIS REPORT")
    print("=" * 80)
    
    # Metadata
    meta = report["analysis_metadata"]
    print(f"\nSource: {meta['source']}")
    print(f"Analyzed: {meta['analyzed_at']}")
    print(f"Text length: {meta['text_length']} characters")
    print(f"Empirical patterns: {'Available' if meta['empirical_patterns_available'] else 'Not trained'}")
    
    # Structural analysis
    struct = report["structural_analysis"]
    print(f"\n📊 STRUCTURAL ANALYSIS")
    print(f"Paragraphs: {struct['paragraph_count']}")
    print(f"Sentences: {struct['total_sentences']}")
    print(f"Words: {struct['total_words']}")
    print(f"Avg sentence length: {struct['average_sentence_length']:.1f} words")
    print(f"Readability score: {struct['readability_score']:.2f}")
    
    # Content analysis
    content = report["content_analysis"]
    print(f"\n🧠 CONTENT ANALYSIS")
    print(f"Neuroscience terms: {len(content['neuroscience_terms'])}")
    print(f"Term density: {content['neuroscience_density']:.2f} terms/sentence")
    if verbose:
        print(f"Key concepts: {', '.join(content['key_concepts'])}")
    
    # Empirical patterns
    emp = report["empirical_pattern_analysis"]
    print(f"\n🔬 EMPIRICAL PATTERN ANALYSIS")
    print(f"Patterns detected: {emp['patterns_detected']}")
    print(f"Statistically significant: {emp['statistically_significant']}")
    
    if emp["pattern_details"]:
        print("\nEmpirical Evidence:")
        for pattern in emp["pattern_details"][:3]:  # Show top 3
            print(f"  • {pattern['description']}")
            print(f"    Validation: {pattern['validation_score']:.2f} | "
                  f"Sample: {pattern['sample_size']} papers")
    
    # Recommendations
    print(f"\n💡 EMPIRICAL RECOMMENDATIONS")
    for i, rec in enumerate(report["empirical_recommendations"][:5], 1):
        print(f"{i}. {rec}")
    
    # Quality assessment
    quality = report["quality_assessment"]
    print(f"\n✅ QUALITY ASSESSMENT")
    print(f"Overall score: {quality['overall_score']:.3f}")
    print(f"Structural quality: {quality['structural_quality']:.3f}")
    
    print("\n" + "=" * 80)
    print("Analysis based on empirical evidence from published neuroscience literature")
    print("=" * 80)


def main():
    """Main entry point for introduction analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze introduction text using empirically-trained patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze text file with trained patterns
  python scripts/analyze_introduction.py --text intro.txt --patterns data/empirical_patterns/
  
  # Analyze with detailed output and save report
  python scripts/analyze_introduction.py --text intro.txt --patterns data/empirical_patterns/ \\
    --report analysis_report.json --verbose
  
  # Analyze direct text input
  python scripts/analyze_introduction.py --text "Working memory represents..." \\
    --patterns data/empirical_patterns/

Requirements:
  - Trained empirical patterns (run scripts/collect_empirical_data.py first)
  - Or train from Zotero: scripts/train_from_zotero.py
        """
    )
    
    parser.add_argument(
        '--text',
        required=True,
        help='Text file to analyze or direct text content'
    )
    
    parser.add_argument(
        '--patterns',
        default='data/empirical_patterns',
        help='Directory containing trained empirical patterns'
    )
    
    parser.add_argument(
        '--report',
        help='File to save detailed analysis report (JSON format)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable detailed output'
    )
    
    args = parser.parse_args()
    
    try:
        # Check if patterns directory exists
        if not Path(args.patterns).exists():
            print(f"❌ Patterns directory not found: {args.patterns}")
            print("Train empirical patterns first:")
            print(f"  python scripts/collect_empirical_data.py --input data/training_papers/")
            print(f"  or python scripts/train_from_zotero.py --collection 'Training Papers'")
            sys.exit(1)
        
        # Run analysis
        result = analyze_introduction(
            text_input=args.text,
            patterns_directory=args.patterns,
            output_file=args.report,
            verbose=args.verbose
        )
        
        print(f"\n✅ Analysis completed successfully")
        if args.report:
            print(f"📄 Detailed report saved to: {args.report}")
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()