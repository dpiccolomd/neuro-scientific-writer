#!/usr/bin/env python3
"""
Citation-Aware Introduction Generator

Generates introduction drafts that integrate specific reference papers
from Zotero collections with empirically-trained writing patterns.

This script provides the complete workflow for:
1. Analyzing reference papers from a Zotero collection
2. Creating citation integration plans based on empirical patterns
3. Generating introduction drafts with contextually appropriate citations
4. Validating citation coherence and scientific accuracy

Usage:
    python scripts/generate_introduction_with_citations.py \\
        --study_title "fMRI Study of Working Memory in Aging" \\
        --research_type "experimental_study" \\
        --domain "cognitive_neuroscience" \\
        --references_collection "My Study References" \\
        --patterns data/empirical_patterns/ \\
        --output introduction_draft.md

Advanced Usage:
    # Multiple citation strategies
    python scripts/generate_introduction_with_citations.py \\
        --multiple_strategies \\
        --citation_styles "problem_gap_solution,hypothesis_driven"
    
    # Interactive mode with validation
    python scripts/generate_introduction_with_citations.py \\
        --interactive \\
        --validate_coherence
"""

import argparse
import logging
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from draft_generator import (
    CitationAwareGenerator, ReferenceIntegrator, DraftValidator
)
from draft_generator.models import (
    StudySpecification, CitationStrategy, ReferenceRole, GeneratedDraft
)
from citation_manager.zotero_integration import ZoteroClient, ZoteroConfig
from citation_manager.models import Reference
from pdf_processor import PDFExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_zotero_client() -> ZoteroClient:
    """Setup Zotero client from environment variables."""
    api_key = os.getenv('ZOTERO_API_KEY')
    user_id = os.getenv('ZOTERO_USER_ID')
    
    if not api_key:
        raise ValueError("Zotero API key required. Set ZOTERO_API_KEY environment variable")
    
    if not user_id:
        raise ValueError("Zotero User ID required. Set ZOTERO_USER_ID environment variable")
    
    config = ZoteroConfig(api_key=api_key, user_id=user_id)
    return ZoteroClient(config)


def collect_references_from_zotero(
    zotero_client: ZoteroClient,
    collection_name: str,
    max_references: int = 15
) -> List[Reference]:
    """Collect references from specified Zotero collection."""
    logger.info(f"Collecting references from Zotero collection: {collection_name}")
    
    try:
        # Get collection items
        items = zotero_client.get_collection_items(collection_name, limit=max_references * 2)
        
        # Convert to Reference objects
        references = []
        for item in items:
            if item.item_type in ['journalArticle', 'conferencePaper', 'book']:
                reference = Reference(
                    title=item.title,
                    authors=[item.authors] if item.authors else [],
                    publication_year=item.publication_year,
                    journal=item.journal,
                    doi=item.doi,
                    abstract=item.abstract,
                    url=item.url
                )
                references.append(reference)
        
        logger.info(f"Collected {len(references)} references from Zotero")
        return references[:max_references]
        
    except Exception as e:
        logger.error(f"Failed to collect references from Zotero: {e}")
        raise


def create_study_specification_interactive() -> StudySpecification:
    """Create study specification through interactive prompts."""
    print("\n" + "="*60)
    print("STUDY SPECIFICATION")
    print("="*60)
    
    study_title = input("Study Title: ").strip()
    
    print("\nResearch Types:")
    research_types = ['clinical_trial', 'experimental_study', 'observational_study', 
                     'systematic_review', 'case_study']
    for i, rtype in enumerate(research_types, 1):
        print(f"  {i}. {rtype.replace('_', ' ').title()}")
    
    research_type_idx = int(input("Select research type (1-5): ")) - 1
    research_type = research_types[research_type_idx]
    
    print("\\nResearch Domains:")
    domains = ['neurosurgery', 'cognitive_neuroscience', 'neuroimaging', 
              'neuro_oncology', 'clinical_neuroscience']
    for i, domain in enumerate(domains, 1):
        print(f"  {i}. {domain.replace('_', ' ').title()}")
    
    domain_idx = int(input("Select research domain (1-5): ")) - 1
    research_domain = domains[domain_idx]
    
    primary_research_question = input("\\nPrimary Research Question: ").strip()
    primary_hypothesis = input("Primary Hypothesis: ").strip()
    
    print("\\nStudy Objectives (enter one per line, empty line to finish):")
    study_objectives = []
    while True:
        objective = input("  - ").strip()
        if not objective:
            break
        study_objectives.append(objective)
    
    target_population = input("\\nTarget Population: ").strip()
    methodology_summary = input("Methodology Summary: ").strip()
    
    print("\\nExpected Outcomes (enter one per line, empty line to finish):")
    expected_outcomes = []
    while True:
        outcome = input("  - ").strip()
        if not outcome:
            break
        expected_outcomes.append(outcome)
    
    clinical_significance = input("\\nClinical Significance: ").strip()
    target_journal = input("Target Journal (optional): ").strip() or None
    
    return StudySpecification(
        study_title=study_title,
        research_type=research_type,
        research_domain=research_domain,
        primary_research_question=primary_research_question,
        primary_hypothesis=primary_hypothesis,
        study_objectives=study_objectives,
        target_population=target_population,
        methodology_summary=methodology_summary,
        expected_outcomes=expected_outcomes,
        clinical_significance=clinical_significance,
        target_journal=target_journal
    )


def create_study_specification_from_args(args) -> StudySpecification:
    """Create study specification from command line arguments."""
    return StudySpecification(
        study_title=args.study_title,
        research_type=args.research_type,
        research_domain=args.domain,
        primary_research_question=args.research_question or f"To investigate {args.study_title}",
        primary_hypothesis=args.hypothesis or f"We hypothesize that the study will reveal significant insights into {args.domain.replace('_', ' ')}",
        study_objectives=args.objectives or [f"Primary objective for {args.study_title}"],
        target_population=args.target_population or "Adult participants",
        methodology_summary=args.methodology or f"{args.research_type.replace('_', ' ')} methodology",
        expected_outcomes=args.expected_outcomes or ["Significant findings expected"],
        clinical_significance=args.clinical_significance or f"This research addresses important questions in {args.domain.replace('_', ' ')}",
        target_journal=args.target_journal
    )


def generate_single_introduction(
    study_spec: StudySpecification,
    references: List[Reference],
    citation_strategy: CitationStrategy,
    empirical_patterns_dir: str,
    target_word_count: int = 400
) -> GeneratedDraft:
    """Generate a single introduction draft."""
    logger.info(f"Generating introduction with {citation_strategy.value} strategy")
    
    # Initialize components
    reference_integrator = ReferenceIntegrator()
    citation_generator = CitationAwareGenerator(empirical_patterns_dir)
    
    # Analyze references
    reference_analyses = reference_integrator.analyze_references_for_study(
        references, study_spec
    )
    
    # Create reference contexts
    reference_contexts = reference_integrator.create_reference_contexts(
        reference_analyses, study_spec
    )
    
    logger.info(f"Created {len(reference_contexts)} reference contexts")
    
    # Generate citation-aware introduction
    draft = citation_generator.generate_citation_aware_introduction(
        study_spec, reference_contexts, citation_strategy, target_word_count
    )
    
    return draft


def generate_multiple_introductions(
    study_spec: StudySpecification,
    references: List[Reference],
    citation_strategies: List[CitationStrategy],
    empirical_patterns_dir: str,
    target_word_count: int = 400
) -> List[GeneratedDraft]:
    """Generate multiple introduction drafts with different strategies."""
    drafts = []
    
    for strategy in citation_strategies:
        try:
            draft = generate_single_introduction(
                study_spec, references, strategy, empirical_patterns_dir, target_word_count
            )
            drafts.append(draft)
        except Exception as e:
            logger.error(f"Failed to generate draft with {strategy.value} strategy: {e}")
    
    return drafts


def save_draft_output(draft: GeneratedDraft, output_file: str, include_metadata: bool = True):
    """Save generated draft to file."""
    output_path = Path(output_file)
    
    # Create content
    content = []
    
    if include_metadata:
        content.append("# Generated Introduction Draft")
        content.append(f"**Study:** {draft.study_specification.study_title}")
        content.append(f"**Strategy:** {draft.integration_plan.strategy.value}")
        content.append(f"**Generated:** {draft.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"**Quality Score:** {draft.quality_scores.get('overall', 0.0):.3f}")
        content.append(f"**Citations Used:** {len(draft.citations_used)}/{len(draft.integration_plan.reference_contexts)}")
        content.append("")
        content.append("---")
        content.append("")
    
    # Add the actual introduction text
    content.append("## Introduction")
    content.append("")
    content.append(draft.generated_text)
    
    if include_metadata:
        content.append("")
        content.append("---")
        content.append("")
        content.append("## Validation Summary")
        content.append(f"- **Citation Completeness:** {draft.citation_completeness:.1%}")
        content.append(f"- **Word Count:** {draft.word_count}")
        content.append(f"- **Paragraphs:** {len(draft.paragraph_breakdown)}")
        
        if draft.citations_missing:
            content.append("")
            content.append("### Missing Citations:")
            for citation in draft.citations_missing:
                content.append(f"- {citation}")
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\\n'.join(content))
    
    logger.info(f"Draft saved to: {output_path}")


def save_detailed_report(draft: GeneratedDraft, validation_report, output_dir: str):
    """Save detailed report with validation results."""
    report_path = Path(output_dir) / f"validation_report_{draft.draft_id}.json"
    
    report_data = {
        "draft_id": draft.draft_id,
        "study_title": draft.study_specification.study_title,
        "generation_metadata": {
            "strategy": draft.integration_plan.strategy.value,
            "generated_at": draft.generated_at.isoformat(),
            "word_count": draft.word_count,
            "paragraphs": len(draft.paragraph_breakdown),
            "references_planned": len(draft.integration_plan.reference_contexts),
            "citations_used": len(draft.citations_used),
            "citations_missing": len(draft.citations_missing)
        },
        "quality_scores": draft.quality_scores,
        "validation_results": {
            "overall_quality": validation_report.overall_quality,
            "empirical_pattern_compliance": validation_report.empirical_pattern_compliance,
            "citation_coherence": validation_report.citation_coherence.overall_coherence,
            "factual_accuracy": validation_report.factual_accuracy,
            "ready_for_submission": validation_report.ready_for_submission
        },
        "issues_and_recommendations": {
            "critical_issues": validation_report.critical_issues,
            "minor_issues": validation_report.minor_issues,
            "recommendations": validation_report.recommendations
        },
        "empirical_validation": draft.empirical_validation
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Detailed report saved to: {report_path}")


def print_draft_summary(draft: GeneratedDraft, validation_report=None):
    """Print summary of generated draft."""
    print("\\n" + "="*80)
    print("CITATION-AWARE INTRODUCTION DRAFT")
    print("="*80)
    
    print(f"Study: {draft.study_specification.study_title}")
    print(f"Strategy: {draft.integration_plan.strategy.value.replace('_', ' ').title()}")
    print(f"Generated: {draft.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\\n📊 DRAFT METRICS")
    print(f"Word Count: {draft.word_count}")
    print(f"Paragraphs: {len(draft.paragraph_breakdown)}")
    print(f"Citations Used: {len(draft.citations_used)}/{len(draft.integration_plan.reference_contexts)}")
    print(f"Citation Completeness: {draft.citation_completeness:.1%}")
    print(f"Quality Score: {draft.quality_scores.get('overall', 0.0):.3f}")
    
    if validation_report:
        print(f"\\n🔍 VALIDATION RESULTS")
        print(f"Overall Quality: {validation_report.overall_quality:.3f}")
        print(f"Citation Coherence: {validation_report.citation_coherence.overall_coherence:.3f}")
        print(f"Ready for Submission: {'✅ Yes' if validation_report.ready_for_submission else '❌ No'}")
        
        if validation_report.critical_issues:
            print(f"\\n🚨 CRITICAL ISSUES:")
            for issue in validation_report.critical_issues[:3]:
                print(f"  • {issue}")
        
        if validation_report.recommendations:
            print(f"\\n💡 RECOMMENDATIONS:")
            for rec in validation_report.recommendations[:3]:
                print(f"  • {rec}")
    
    print(f"\\n📝 GENERATED TEXT:")
    print("-" * 80)
    print(draft.generated_text)
    print("-" * 80)


def main():
    """Main entry point for citation-aware introduction generation."""
    parser = argparse.ArgumentParser(
        description="Generate citation-aware introduction drafts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/generate_introduction_with_citations.py \\
    --study_title "fMRI Study of Working Memory in Aging" \\
    --research_type experimental_study \\
    --domain cognitive_neuroscience \\
    --references_collection "Study References"
  
  # With custom parameters
  python scripts/generate_introduction_with_citations.py \\
    --study_title "Deep Brain Stimulation for Depression" \\
    --research_type clinical_trial \\
    --domain neurosurgery \\
    --references_collection "DBS References" \\
    --target_journal "Nature Neuroscience" \\
    --word_count 500 \\
    --output dbs_introduction.md
  
  # Multiple strategies comparison
  python scripts/generate_introduction_with_citations.py \\
    --multiple_strategies \\
    --citation_styles problem_gap_solution hypothesis_driven \\
    --output_dir results/
  
  # Interactive mode
  python scripts/generate_introduction_with_citations.py \\
    --interactive \\
    --validate_coherence
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--study_title',
        help='Title of the research study'
    )
    
    parser.add_argument(
        '--research_type',
        choices=['clinical_trial', 'experimental_study', 'observational_study', 
                'systematic_review', 'case_study'],
        help='Type of research study'
    )
    
    parser.add_argument(
        '--domain',
        help='Research domain (e.g., neurosurgery, cognitive_neuroscience)'
    )
    
    parser.add_argument(
        '--references_collection',
        required=True,
        help='Name of Zotero collection containing study references'
    )
    
    # Optional study specification
    parser.add_argument(
        '--research_question',
        help='Primary research question'
    )
    
    parser.add_argument(
        '--hypothesis',
        help='Primary research hypothesis'
    )
    
    parser.add_argument(
        '--objectives',
        nargs='+',
        help='Study objectives'
    )
    
    parser.add_argument(
        '--target_population',
        help='Target study population'
    )
    
    parser.add_argument(
        '--methodology',
        help='Methodology summary'
    )
    
    parser.add_argument(
        '--expected_outcomes',
        nargs='+',
        help='Expected study outcomes'
    )
    
    parser.add_argument(
        '--clinical_significance',
        help='Clinical significance of the study'
    )
    
    parser.add_argument(
        '--target_journal',
        help='Target journal for publication'
    )
    
    # Generation parameters
    parser.add_argument(
        '--patterns',
        default='data/empirical_patterns',
        help='Directory containing empirical patterns'
    )
    
    parser.add_argument(
        '--citation_strategy',
        choices=['problem_gap_solution', 'hypothesis_driven', 'methodology_focused', 'literature_synthesis'],
        default='problem_gap_solution',
        help='Citation integration strategy'
    )
    
    parser.add_argument(
        '--word_count',
        type=int,
        default=400,
        help='Target word count for introduction'
    )
    
    parser.add_argument(
        '--max_references',
        type=int,
        default=15,
        help='Maximum number of references to include'
    )
    
    # Multiple strategies
    parser.add_argument(
        '--multiple_strategies',
        action='store_true',
        help='Generate drafts with multiple citation strategies'
    )
    
    parser.add_argument(
        '--citation_styles',
        nargs='+',
        choices=['problem_gap_solution', 'hypothesis_driven', 'methodology_focused'],
        default=['problem_gap_solution', 'hypothesis_driven'],
        help='Citation strategies for multiple draft generation'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        help='Output file for generated introduction'
    )
    
    parser.add_argument(
        '--output_dir',
        default='results',
        help='Output directory for multiple drafts'
    )
    
    parser.add_argument(
        '--save_detailed_report',
        action='store_true',
        help='Save detailed validation report'
    )
    
    # Interactive and validation
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode for study specification'
    )
    
    parser.add_argument(
        '--validate_coherence',
        action='store_true',
        help='Perform citation coherence validation'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Check empirical patterns
        if not Path(args.patterns).exists():
            print(f"❌ Empirical patterns not found: {args.patterns}")
            print("Train empirical patterns first:")
            print("  PYTHONPATH=./src python scripts/train_from_zotero.py --collection 'Training Papers'")
            sys.exit(1)
        
        # Setup Zotero client
        print("🔗 Connecting to Zotero...")
        zotero_client = setup_zotero_client()
        
        # Collect references
        print(f"📚 Collecting references from '{args.references_collection}'...")
        references = collect_references_from_zotero(
            zotero_client, args.references_collection, args.max_references
        )
        
        if not references:
            print(f"❌ No references found in collection: {args.references_collection}")
            sys.exit(1)
        
        # Create study specification
        if args.interactive:
            study_spec = create_study_specification_interactive()
        else:
            if not all([args.study_title, args.research_type, args.domain]):
                print("❌ Required arguments missing: --study_title, --research_type, --domain")
                print("Use --interactive for interactive mode")
                sys.exit(1)
            study_spec = create_study_specification_from_args(args)
        
        # Generate introduction(s)
        if args.multiple_strategies:
            print("🔄 Generating multiple introduction drafts...")
            strategies = [CitationStrategy(s) for s in args.citation_styles]
            drafts = generate_multiple_introductions(
                study_spec, references, strategies, args.patterns, args.word_count
            )
            
            # Save multiple drafts
            Path(args.output_dir).mkdir(exist_ok=True)
            
            for i, draft in enumerate(drafts):
                output_file = Path(args.output_dir) / f"introduction_{draft.integration_plan.strategy.value}.md"
                save_draft_output(draft, str(output_file))
                print_draft_summary(draft)
                
                if args.validate_coherence:
                    validator = DraftValidator()
                    validation_report = validator.validate_draft(draft)
                    if args.save_detailed_report:
                        save_detailed_report(draft, validation_report, args.output_dir)
        
        else:
            print("✍️ Generating citation-aware introduction...")
            strategy = CitationStrategy(args.citation_strategy)
            draft = generate_single_introduction(
                study_spec, references, strategy, args.patterns, args.word_count
            )
            
            # Validate if requested
            validation_report = None
            if args.validate_coherence:
                print("🔍 Validating citation coherence...")
                validator = DraftValidator()
                validation_report = validator.validate_draft(draft)
            
            # Save output
            if args.output:
                save_draft_output(draft, args.output)
                print(f"💾 Draft saved to: {args.output}")
            
            if args.save_detailed_report:
                output_dir = Path(args.output).parent if args.output else Path('results')
                output_dir.mkdir(exist_ok=True)
                save_detailed_report(draft, validation_report, str(output_dir))
            
            # Print summary
            print_draft_summary(draft, validation_report)
        
        print("\\n✅ Citation-aware introduction generation complete!")
        
    except KeyboardInterrupt:
        print("\\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Introduction generation failed: {e}")
        print(f"\\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()