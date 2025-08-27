# Zotero Integration Guide 🔬📚

## Overview

The Neuroscience Writing Assistant now includes **complete Zotero integration** for automated training of empirical patterns. You can now train the system directly from your Zotero library with a single command.

## What You Get After Training

### 1. **Empirical Pattern Database**
When you train from Zotero, the system creates:

- **`data/empirical_patterns/empirical_patterns.json`** - Statistical patterns from your papers
- **`data/empirical_patterns/structural_metrics.json`** - Introduction structure analysis  
- **`data/empirical_patterns/journal_profiles.json`** - Journal-specific patterns (if 50+ papers per journal)
- **`data/training_papers/zotero_metadata.json`** - Source paper metadata and DOIs

### 2. **Statistical Validation**
All patterns include:
- **Confidence intervals** with statistical significance testing
- **Sample sizes** for each pattern (minimum 50 papers required)
- **Journal analysis** only when sufficient data exists (no assumptions!)
- **Cross-validation** against independent paper sets

### 3. **Evidence-Based Templates**
Templates generated from your trained patterns will state:
- *"Based on analysis of 73 Nature Neuroscience papers, optimal introduction length is 4.2±0.8 paragraphs"*
- *"Problem-gap-solution structure used in 68% of successful papers (CI: 61%-75%)"*
- *"Empirical evidence from your collection supports conceptual breadth progression"*

## Quick Start

### Step 1: Get Zotero API Access
```bash
# Get your API credentials
python scripts/train_from_zotero.py --setup_help
```

### Step 2: Train from Your Zotero Library
```bash
# Set environment variables (recommended)
export ZOTERO_API_KEY="your_api_key_here"
export ZOTERO_USER_ID="your_user_id_here"

# Train from specific collection
python scripts/train_from_zotero.py --collection "Neuroscience Papers"

# Or train from entire library
python scripts/train_from_zotero.py --min_papers 75
```

### Step 3: Use Your Trained Patterns
```bash
# Analyze an introduction with YOUR patterns
python scripts/analyze_introduction.py --text intro.txt --patterns data/empirical_patterns/

# Generate template based on YOUR evidence
python scripts/generate_template.py --research_type clinical_trial --domain neurosurgery --patterns data/empirical_patterns/
```

## Training Output Example

```
🎉 ZOTERO TRAINING COMPLETED SUCCESSFULLY
================================================================================
📚 Training papers: 67 from your Zotero library
🏛️  Journals: Nature Neuroscience, Neuron, Journal of Neuroscience, NeuroImage, Brain
🔬 Domains: cognitive_neuroscience, neuroimaging, cellular_neuroscience
📊 Empirical patterns: 12 statistically validated
💾 Saved to: data/empirical_patterns

🚀 READY FOR ANALYSIS:
python scripts/analyze_introduction.py --text intro.txt --patterns data/empirical_patterns
```

## Professional Tools (Not Demos!)

### 1. **Real Analysis Tool**
- **`scripts/analyze_introduction.py`** - Uses YOUR empirical patterns (not simulations)
- Provides statistical validation based on your paper collection
- Medical-grade quality scoring with confidence intervals

### 2. **Template Generator** 
- **`scripts/generate_template.py`** - Creates templates from YOUR empirical data
- Journal-specific patterns only if you have 50+ papers from that journal
- No assumptions - only evidence-based recommendations

### 3. **Educational Tutorial**
- **`examples/pattern_comparison_tutorial.py`** - Learn the difference between assumptions vs evidence
- Clearly labeled as educational (not for production use)

## Advanced Usage

### Group Libraries
```bash
# Train from Zotero group library
python scripts/train_from_zotero.py --library_type group --group_id 12345 --api_key YOUR_KEY
```

### Large Collections
```bash
# Process up to 200 papers with detailed logging
python scripts/train_from_zotero.py --max_papers 200 --verbose
```

### Quality Analysis
```bash
# Analyze your trained patterns
python scripts/analyze_introduction.py --text sample_intro.txt --patterns data/empirical_patterns/ --report analysis_report.json --verbose
```

## Requirements

### Minimum for Statistical Validity
- **50+ neuroscience papers** with PDF attachments in Zotero
- **Multiple journals** for broader pattern detection
- **Recent papers** (last 10 years recommended)

### Optimal for Best Results
- **100+ papers** spanning major neuroscience journals
- **Multiple research domains** (cognitive, cellular, clinical)
- **High-quality PDFs** with extractable text

## Output Quality

### What You'll See
```
📊 EMPIRICAL PATTERN ANALYSIS
Patterns detected: 8
Statistically significant: 6

Empirical Evidence:
  • Optimal introduction length: 4.3±0.7 paragraphs (based on 67 papers)
    Validation: 0.82 | Sample: 67 papers
  • Problem-gap-solution structure: Used in 71% of papers (CI: 63%-78%)
    Validation: 0.89 | Sample: 54 papers

💡 EMPIRICAL RECOMMENDATIONS
1. Consider adjusting paragraph count: current 3, empirical optimum 4.3±0.7 (based on 67 papers)
2. Most successful argumentation structure: Problem-gap-solution (used in 71% of papers)
3. Current neuroscience term density: 1.4 terms/sentence. Compare against empirical benchmarks.

✅ QUALITY ASSESSMENT
Overall score: 0.847
Analysis based on empirical evidence from published neuroscience literature
```

## Scientific Integrity

- **No Assumptions**: All recommendations based on your paper collection
- **Statistical Rigor**: Confidence intervals and significance testing
- **Transparent Sources**: Full metadata linking back to original papers
- **Medical-Grade**: Suitable for research publication and clinical use

## Next Steps

1. **Collect 50+ neuroscience papers** in your Zotero library
2. **Run training**: `python scripts/train_from_zotero.py --collection "Training Papers"`
3. **Analyze your writing**: Use the trained patterns for real analysis
4. **Generate templates**: Create evidence-based introduction templates

**No more guesswork - only empirical evidence from your curated literature!** 🔬✅