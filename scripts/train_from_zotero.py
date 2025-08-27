#!/usr/bin/env python3
"""
Zotero Training Script

One-command training of empirical patterns from Zotero library.
Connects to your Zotero account, downloads neuroscience papers,
and trains empirical patterns for scientific writing analysis.

Usage:
    python scripts/train_from_zotero.py --collection "Neuroscience Papers" --api_key YOUR_KEY
    python scripts/train_from_zotero.py --user_id 123456 --api_key YOUR_KEY --min_papers 75
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from citation_manager.zotero_integration import ZoteroConfig, ZoteroTrainingManager
from citation_manager.exceptions import ZoteroIntegrationError
from pdf_processor import PDFExtractor
from analysis.empirical_pattern_detector import EmpiricalPatternDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_zotero_config(
    api_key: str,
    user_id: Optional[str] = None,
    group_id: Optional[str] = None,
    library_type: str = "user"
) -> ZoteroConfig:
    """Setup Zotero configuration from arguments or environment."""
    
    # Try environment variables if not provided
    if not api_key:
        api_key = os.getenv('ZOTERO_API_KEY')
        if not api_key:
            raise ValueError("Zotero API key required. Use --api_key or set ZOTERO_API_KEY environment variable")
    
    if library_type == "user" and not user_id:
        user_id = os.getenv('ZOTERO_USER_ID')
        if not user_id:
            raise ValueError("User ID required for user library. Use --user_id or set ZOTERO_USER_ID environment variable")
    
    if library_type == "group" and not group_id:
        group_id = os.getenv('ZOTERO_GROUP_ID')
        if not group_id:
            raise ValueError("Group ID required for group library. Use --group_id or set ZOTERO_GROUP_ID environment variable")
    
    return ZoteroConfig(
        api_key=api_key,
        user_id=user_id,
        group_id=group_id,
        library_type=library_type
    )


def train_from_zotero(
    zotero_config: ZoteroConfig,
    collection_name: Optional[str] = None,
    min_papers: int = 50,
    max_papers: int = 200,
    output_dir: str = "data/training_papers",
    patterns_dir: str = "data/empirical_patterns"
) -> dict:
    """
    Complete training workflow from Zotero to empirical patterns.
    
    Args:
        zotero_config: Zotero API configuration
        collection_name: Specific collection to use (None = entire library)
        min_papers: Minimum papers required for training
        max_papers: Maximum papers to process
        output_dir: Directory to save PDFs
        patterns_dir: Directory to save empirical patterns
        
    Returns:
        Training results dictionary
    """
    print("=" * 80)
    print("NEUROSCIENCE WRITING ASSISTANT - ZOTERO TRAINING")
    print("=" * 80)
    print("🔬 Training empirical patterns from your Zotero library")
    print("📚 This replaces assumptions with evidence from published literature")
    print()
    
    # Phase 1: Collect papers from Zotero
    logger.info("Phase 1: Collecting papers from Zotero")
    print("📥 PHASE 1: COLLECTING PAPERS FROM ZOTERO")
    print("-" * 50)
    
    training_manager = ZoteroTrainingManager(zotero_config, output_dir)
    
    try:
        collection_results = training_manager.collect_training_papers(
            collection_name=collection_name,
            min_papers=min_papers,
            max_papers=max_papers
        )
        
        print(f"✅ Papers collected: {collection_results['successful_downloads']}")
        print(f"📊 Total items scanned: {collection_results['total_items_found']}")
        print(f"🧠 Neuroscience papers found: {collection_results['neuroscience_papers']}")
        print(f"⚠️  Failed downloads: {collection_results['failed_downloads']}")
        
    except ZoteroIntegrationError as e:
        print(f"❌ Zotero collection failed: {e}")
        return {"success": False, "error": str(e)}
    
    # Show collection summary
    summary = training_manager.get_collection_summary()
    print(f"\n📈 COLLECTION SUMMARY:")
    print(f"Total PDFs: {summary['total_pdfs']}")
    print(f"Journals represented: {summary['journals_represented']}")
    print(f"Top journals: {', '.join(list(summary['top_journals'].keys())[:5])}")
    
    # Phase 2: Process PDFs and extract content
    logger.info("Phase 2: Processing PDFs")
    print(f"\n🔍 PHASE 2: PROCESSING PDFS")
    print("-" * 50)
    
    pdf_extractor = PDFExtractor()
    pdf_files = list(Path(output_dir).glob("*.pdf"))
    processed_documents = []
    
    successful_extractions = 0
    failed_extractions = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"Processing {i}/{len(pdf_files)}: {pdf_file.name}")
        
        try:
            doc = pdf_extractor.process_pdf(str(pdf_file))
            if doc and doc.introduction_section:
                processed_documents.append(doc)
                successful_extractions += 1
                print(f"  ✅ Introduction extracted ({len(doc.introduction_section.content)} chars)")
            else:
                failed_extractions += 1
                print(f"  ⚠️  No introduction found")
        except Exception as e:
            failed_extractions += 1
            print(f"  ❌ Processing failed: {e}")
    
    print(f"\n📊 PDF Processing Results:")
    print(f"Successful extractions: {successful_extractions}")
    print(f"Failed extractions: {failed_extractions}")
    print(f"Success rate: {successful_extractions/len(pdf_files):.1%}")
    
    if successful_extractions < min_papers:
        error_msg = (f"Insufficient successful extractions: {successful_extractions} < {min_papers}. "
                    "Need higher quality PDFs with extractable introduction sections.")
        print(f"❌ {error_msg}")
        return {"success": False, "error": error_msg}
    
    # Phase 3: Train empirical patterns
    logger.info("Phase 3: Training empirical patterns")
    print(f"\n🧠 PHASE 3: TRAINING EMPIRICAL PATTERNS")
    print("-" * 50)
    print("🔬 Performing statistical analysis of introduction structures...")
    
    pattern_detector = EmpiricalPatternDetector(patterns_dir)
    
    try:
        training_result = pattern_detector.collect_empirical_data(
            processed_documents,
            min_sample_size=min_papers
        )
        
        print(f"✅ Empirical training completed!")
        print(f"📊 Papers analyzed: {training_result.successful_extractions}")
        print(f"🏛️  Journals covered: {len(training_result.journals_covered)}")
        print(f"🔬 Research domains: {len(training_result.domains_covered)}")
        print(f"📈 Patterns identified: {len(training_result.patterns_identified)}")
        
        # Show significant patterns
        significant_patterns = [p for p in training_result.patterns_identified if p.is_statistically_significant]
        print(f"📊 Statistically significant patterns: {len(significant_patterns)}")
        
        if significant_patterns:
            print(f"\n🎯 KEY EMPIRICAL FINDINGS:")
            for pattern in significant_patterns[:3]:  # Show top 3
                print(f"  • {pattern.description}")
                print(f"    Sample: {pattern.sample_size} papers, Confidence: {pattern.validation_score:.2f}")
        
    except Exception as e:
        error_msg = f"Empirical pattern training failed: {e}"
        print(f"❌ {error_msg}")
        return {"success": False, "error": error_msg}
    
    # Final summary
    print(f"\n" + "=" * 80)
    print("🎉 ZOTERO TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"📚 Training papers: {successful_extractions} from your Zotero library")
    print(f"🏛️  Journals: {', '.join(training_result.journals_covered[:5])}")
    print(f"🔬 Domains: {', '.join(training_result.domains_covered)}")
    print(f"📊 Empirical patterns: {len(significant_patterns)} statistically validated")
    print(f"💾 Saved to: {patterns_dir}")
    
    print(f"\n🚀 READY FOR ANALYSIS:")
    print(f"python scripts/analyze_introduction.py --text intro.txt --patterns {patterns_dir}")
    print(f"python scripts/generate_template.py --research_type clinical_trial --patterns {patterns_dir}")
    
    return {
        "success": True,
        "papers_collected": collection_results['successful_downloads'],
        "papers_processed": successful_extractions,
        "patterns_identified": len(training_result.patterns_identified),
        "significant_patterns": len(significant_patterns),
        "journals_covered": training_result.journals_covered,
        "domains_covered": training_result.domains_covered,
        "patterns_directory": patterns_dir
    }


def print_setup_instructions():
    """Print instructions for setting up Zotero API access."""
    print("=" * 80)
    print("ZOTERO API SETUP INSTRUCTIONS")
    print("=" * 80)
    print("1. Get your Zotero API key:")
    print("   • Go to https://www.zotero.org/settings/keys")
    print("   • Click 'Create new private key'")
    print("   • Check 'Allow library access'")
    print("   • Copy the generated key")
    print("")
    print("2. Find your User ID:")
    print("   • Go to https://www.zotero.org/settings/keys")
    print("   • Your User ID is shown at the top of the page")
    print("")
    print("3. Set environment variables (optional):")
    print("   export ZOTERO_API_KEY='your_api_key_here'")
    print("   export ZOTERO_USER_ID='your_user_id_here'")
    print("")
    print("4. For group libraries:")
    print("   • Get Group ID from group settings page")
    print("   • Use --library_type group --group_id GROUP_ID")
    print("=" * 80)


def main():
    """Main entry point for Zotero training."""
    parser = argparse.ArgumentParser(
        description="Train empirical patterns from Zotero library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from specific collection
  python scripts/train_from_zotero.py --collection "Neuroscience Papers" \\
    --api_key YOUR_KEY --user_id YOUR_ID
  
  # Train from entire library with environment variables
  export ZOTERO_API_KEY="your_key"
  export ZOTERO_USER_ID="your_id"
  python scripts/train_from_zotero.py --min_papers 75
  
  # Train from group library
  python scripts/train_from_zotero.py --library_type group --group_id 12345 \\
    --api_key YOUR_KEY --collection "Research Papers"

Requirements:
  - Zotero account with API access
  - 50+ neuroscience papers with PDF attachments
  - Papers should span multiple journals for best results
        """
    )
    
    parser.add_argument(
        '--api_key',
        help='Zotero API key (or set ZOTERO_API_KEY env var)'
    )
    
    parser.add_argument(
        '--user_id',
        help='Zotero user ID (or set ZOTERO_USER_ID env var)'
    )
    
    parser.add_argument(
        '--group_id',
        help='Zotero group ID for group libraries'
    )
    
    parser.add_argument(
        '--library_type',
        choices=['user', 'group'],
        default='user',
        help='Library type: user or group (default: user)'
    )
    
    parser.add_argument(
        '--collection',
        help='Specific collection name to use (default: entire library)'
    )
    
    parser.add_argument(
        '--min_papers',
        type=int,
        default=50,
        help='Minimum papers required for training (default: 50)'
    )
    
    parser.add_argument(
        '--max_papers',
        type=int,
        default=200,
        help='Maximum papers to process (default: 200)'
    )
    
    parser.add_argument(
        '--output_dir',
        default='data/training_papers',
        help='Directory to save PDFs (default: data/training_papers)'
    )
    
    parser.add_argument(
        '--patterns_dir',
        default='data/empirical_patterns',
        help='Directory to save patterns (default: data/empirical_patterns)'
    )
    
    parser.add_argument(
        '--setup_help',
        action='store_true',
        help='Show Zotero API setup instructions'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.setup_help:
        print_setup_instructions()
        return
    
    try:
        # Setup Zotero configuration
        config = setup_zotero_config(
            api_key=args.api_key,
            user_id=args.user_id,
            group_id=args.group_id,
            library_type=args.library_type
        )
        
        # Run training
        result = train_from_zotero(
            zotero_config=config,
            collection_name=args.collection,
            min_papers=args.min_papers,
            max_papers=args.max_papers,
            output_dir=args.output_dir,
            patterns_dir=args.patterns_dir
        )
        
        if result["success"]:
            print(f"\n✅ Training completed successfully!")
            print(f"🔬 {result['significant_patterns']} statistically significant patterns identified")
            print(f"📊 Ready for scientific analysis with empirical validation")
        else:
            print(f"\n❌ Training failed: {result['error']}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n❌ Error: {e}")
        print(f"\nFor setup help, run:")
        print(f"python scripts/train_from_zotero.py --setup_help")
        sys.exit(1)


if __name__ == "__main__":
    main()