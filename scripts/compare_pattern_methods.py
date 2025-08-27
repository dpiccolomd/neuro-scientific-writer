#!/usr/bin/env python3
"""
Pattern Detection Comparison Tool

Compare rule-based (naive) pattern detection with empirical pattern detection
to demonstrate the improvement in accuracy and scientific rigor.

Usage:
    python scripts/compare_pattern_methods.py --input data/test_papers/ --empirical_data data/empirical_patterns/
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pdf_processor import PDFExtractor
from analysis.text_analyzer import NeuroTextAnalyzer
from analysis.pattern_detector import WritingPatternDetector  # Rule-based
from analysis.empirical_pattern_detector import EmpiricalPatternDetector  # Empirical

logger = logging.getLogger(__name__)


def compare_pattern_detection_methods(
    test_papers_dir: str,
    empirical_data_dir: str,
    output_file: str = "pattern_comparison_report.json"
) -> Dict[str, Any]:
    """
    Compare rule-based vs empirical pattern detection methods.
    
    Args:
        test_papers_dir: Directory with test papers
        empirical_data_dir: Directory with empirical pattern data
        output_file: Output file for comparison report
        
    Returns:
        Comparison results dictionary
    """
    logger.info("Starting pattern detection method comparison")
    
    # Initialize components
    pdf_extractor = PDFExtractor()
    text_analyzer = NeuroTextAnalyzer()
    rule_based_detector = WritingPatternDetector()
    empirical_detector = EmpiricalPatternDetector(empirical_data_dir)
    
    # Process test papers
    test_papers_path = Path(test_papers_dir)
    pdf_files = list(test_papers_path.glob("*.pdf"))[:10]  # Limit for comparison
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in {test_papers_dir}")
    
    logger.info(f"Processing {len(pdf_files)} test papers for comparison")
    
    comparison_results = {
        "methodology": {
            "rule_based": {
                "description": "Terminology-focused, assumption-based pattern detection",
                "approach": "Hardcoded thresholds and keyword counting",
                "empirical_foundation": False
            },
            "empirical": {
                "description": "Statistically derived patterns from published literature",
                "approach": "Data-driven analysis of successful publications",
                "empirical_foundation": True
            }
        },
        "test_cases": [],
        "summary": {}
    }
    
    rule_based_patterns_all = []
    empirical_patterns_all = []
    
    for i, pdf_file in enumerate(pdf_files, 1):
        logger.info(f"Analyzing {i}/{len(pdf_files)}: {pdf_file.name}")
        
        try:
            # Extract and analyze document
            doc = pdf_extractor.process_pdf(str(pdf_file))
            if not doc or not doc.introduction_section:
                continue
            
            # Analyze text
            analysis = text_analyzer.analyze_text(
                doc.introduction_section.content,
                f"test_{i}",
                "introduction"
            )
            
            # Rule-based pattern detection
            rule_based_patterns = rule_based_detector.detect_patterns([analysis])
            
            # Empirical pattern detection
            empirical_patterns = empirical_detector.detect_patterns_empirical([analysis])
            
            # Store results
            test_case = {
                "document": pdf_file.name,
                "rule_based": {
                    "patterns_detected": len(rule_based_patterns),
                    "pattern_types": [p.pattern_type for p in rule_based_patterns],
                    "confidence_scores": [p.confidence for p in rule_based_patterns],
                    "avg_confidence": sum(p.confidence for p in rule_based_patterns) / len(rule_based_patterns) if rule_based_patterns else 0
                },
                "empirical": {
                    "patterns_detected": len(empirical_patterns),
                    "pattern_types": [p.pattern_type for p in empirical_patterns],
                    "validation_scores": [p.validation_score for p in empirical_patterns],
                    "avg_validation": sum(p.validation_score for p in empirical_patterns) / len(empirical_patterns) if empirical_patterns else 0,
                    "statistically_significant": sum(1 for p in empirical_patterns if p.is_statistically_significant)
                }
            }
            
            comparison_results["test_cases"].append(test_case)
            rule_based_patterns_all.extend(rule_based_patterns)
            empirical_patterns_all.extend(empirical_patterns)
            
        except Exception as e:
            logger.warning(f"Failed to analyze {pdf_file.name}: {e}")
    
    # Calculate summary statistics
    summary = calculate_comparison_summary(
        rule_based_patterns_all, 
        empirical_patterns_all, 
        comparison_results["test_cases"]
    )
    comparison_results["summary"] = summary
    
    # Save results
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    logger.info(f"Comparison results saved to {output_path}")
    
    return comparison_results


def calculate_comparison_summary(
    rule_based_patterns: List,
    empirical_patterns: List,
    test_cases: List[Dict]
) -> Dict[str, Any]:
    """Calculate summary statistics for pattern detection comparison."""
    
    if not test_cases:
        return {"error": "No test cases processed"}
    
    # Rule-based statistics
    rule_based_stats = {
        "total_patterns": len(rule_based_patterns),
        "avg_patterns_per_document": len(rule_based_patterns) / len(test_cases),
        "avg_confidence": sum(tc["rule_based"]["avg_confidence"] for tc in test_cases) / len(test_cases),
        "pattern_types": list(set(p.pattern_type for p in rule_based_patterns)),
        "limitations": [
            "Based on hardcoded assumptions",
            "No empirical validation",
            "Terminology-focused approach",
            "Cannot adapt to new evidence"
        ]
    }
    
    # Empirical statistics
    empirical_stats = {
        "total_patterns": len(empirical_patterns),
        "avg_patterns_per_document": len(empirical_patterns) / len(test_cases) if empirical_patterns else 0,
        "avg_validation": sum(tc["empirical"]["avg_validation"] for tc in test_cases) / len(test_cases),
        "statistically_significant": sum(tc["empirical"]["statistically_significant"] for tc in test_cases),
        "pattern_types": list(set(p.pattern_type for p in empirical_patterns)) if empirical_patterns else [],
        "advantages": [
            "Statistically validated patterns",
            "Based on analysis of published papers",
            "Adapts to new empirical evidence",
            "Provides confidence intervals"
        ]
    }
    
    # Comparison insights
    insights = [
        f"Rule-based detection found {rule_based_stats['total_patterns']} patterns vs {empirical_stats['total_patterns']} empirical patterns",
        f"Average confidence: Rule-based {rule_based_stats['avg_confidence']:.3f} vs Empirical validation {empirical_stats['avg_validation']:.3f}",
        f"Empirical method identified {empirical_stats['statistically_significant']} statistically significant patterns"
    ]
    
    if empirical_stats['total_patterns'] == 0:
        insights.append("⚠️ No empirical patterns detected - requires trained empirical model from 50+ papers")
    
    return {
        "rule_based": rule_based_stats,
        "empirical": empirical_stats,
        "insights": insights,
        "recommendation": generate_method_recommendation(rule_based_stats, empirical_stats)
    }


def generate_method_recommendation(rule_based_stats: Dict, empirical_stats: Dict) -> str:
    """Generate recommendation on which method to use."""
    
    if empirical_stats['total_patterns'] == 0:
        return (
            "RECOMMENDATION: Complete empirical data collection first. "
            "Current rule-based system is acknowledged as naive and should be replaced "
            "with empirically derived patterns for scientific rigor."
        )
    
    if empirical_stats['statistically_significant'] > rule_based_stats['total_patterns'] * 0.5:
        return (
            "RECOMMENDATION: Use empirical pattern detection. "
            "It provides statistically validated patterns with higher scientific rigor "
            "compared to assumption-based rule detection."
        )
    else:
        return (
            "RECOMMENDATION: Collect more empirical data. "
            "Current empirical model needs larger sample size for robust pattern detection."
        )


def print_comparison_report(results: Dict[str, Any]):
    """Print a formatted comparison report."""
    print("\n" + "=" * 80)
    print("PATTERN DETECTION METHOD COMPARISON REPORT")
    print("=" * 80)
    
    # Methodology comparison
    print("\n📋 METHODOLOGY COMPARISON")
    print("-" * 50)
    
    rule_based = results["methodology"]["rule_based"]
    empirical = results["methodology"]["empirical"]
    
    print(f"Rule-Based Approach:")
    print(f"  Description: {rule_based['description']}")
    print(f"  Method: {rule_based['approach']}")
    print(f"  Empirical Foundation: {rule_based['empirical_foundation']}")
    
    print(f"\nEmpirical Approach:")
    print(f"  Description: {empirical['description']}")
    print(f"  Method: {empirical['approach']}")
    print(f"  Empirical Foundation: {empirical['empirical_foundation']}")
    
    # Results summary
    if "summary" in results and "error" not in results["summary"]:
        summary = results["summary"]
        
        print(f"\n📊 DETECTION RESULTS")
        print("-" * 50)
        
        print(f"Rule-Based Detection:")
        print(f"  Total patterns detected: {summary['rule_based']['total_patterns']}")
        print(f"  Average per document: {summary['rule_based']['avg_patterns_per_document']:.1f}")
        print(f"  Average confidence: {summary['rule_based']['avg_confidence']:.3f}")
        print(f"  Pattern types: {', '.join(summary['rule_based']['pattern_types'])}")
        
        print(f"\nEmpirical Detection:")
        print(f"  Total patterns detected: {summary['empirical']['total_patterns']}")
        print(f"  Average per document: {summary['empirical']['avg_patterns_per_document']:.1f}")
        print(f"  Average validation: {summary['empirical']['avg_validation']:.3f}")
        print(f"  Statistically significant: {summary['empirical']['statistically_significant']}")
        if summary['empirical']['pattern_types']:
            print(f"  Pattern types: {', '.join(summary['empirical']['pattern_types'])}")
        
        print(f"\n🔍 KEY INSIGHTS")
        print("-" * 50)
        for insight in summary['insights']:
            print(f"  • {insight}")
        
        print(f"\n💡 RECOMMENDATION")
        print("-" * 50)
        print(f"  {summary['recommendation']}")
    
    print("\n" + "=" * 80)
    print("SCIENTIFIC RIGOR: Empirical patterns provide statistical validation")
    print("vs rule-based assumptions for neuroscience writing analysis.")
    print("=" * 80)


def main():
    """Main entry point for pattern comparison."""
    parser = argparse.ArgumentParser(
        description="Compare rule-based vs empirical pattern detection methods"
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Directory containing test PDF papers'
    )
    
    parser.add_argument(
        '--empirical_data',
        default='data/empirical_patterns',
        help='Directory with empirical pattern data'
    )
    
    parser.add_argument(
        '--output',
        default='pattern_comparison_report.json',
        help='Output file for comparison report'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        # Run comparison
        results = compare_pattern_detection_methods(
            test_papers_dir=args.input,
            empirical_data_dir=args.empirical_data,
            output_file=args.output
        )
        
        # Print results
        print_comparison_report(results)
        
        print(f"\n✅ Pattern comparison completed")
        print(f"📄 Detailed results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Pattern comparison failed: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()