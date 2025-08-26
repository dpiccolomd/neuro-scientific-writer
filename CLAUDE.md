# Neuro-Scientific Writing Assistant

## Project Overview
AI-powered scientific writing companion specialized in neuroscience/neurosurgery that learns from reference papers to generate structured introduction templates and drafts.

**Target Field**: Neuroscience, Neurosurgery, Neuro-oncology
**Citation Style**: APA (flexible output)
**Goal**: Generate publication-ready introduction drafts with integrated citations

## Project Structure
```
neuro-scientific-writer/
├── src/
│   ├── pdf_processor/          # PDF extraction and parsing
│   ├── analysis/              # NLP and content analysis
│   ├── template_engine/       # Template generation
│   ├── citation_manager/      # Reference handling
│   ├── quality_control/       # Validation and fact-checking
│   └── api/                   # FastAPI backend
├── data/
│   ├── training_papers/       # Reference PDF collection
│   ├── templates/            # Generated templates
│   └── embeddings/           # Vector database
├── frontend/                 # Streamlit interface
├── tests/
├── docs/
└── requirements.txt
```

## Key Commands

### Development
- `python -m pytest tests/` - Run all tests
- `python src/api/main.py` - Start backend server
- `streamlit run frontend/app.py` - Launch web interface
- `python scripts/train_template.py --input data/training_papers/` - Generate templates
- `python scripts/generate_intro.py --project "project_name" --refs data/references/` - Generate introduction

### Quality Checks
- `python -m flake8 src/` - Code linting
- `python -m mypy src/` - Type checking
- `python scripts/validate_accuracy.py` - Check generated content accuracy

## Workflow

### Phase 1: Template Training
1. Collect 20-50 neuroscience papers in `data/training_papers/`
2. Run PDF processing: `python scripts/process_pdfs.py`
3. Analyze writing patterns: `python scripts/analyze_structure.py`
4. Generate template: `python scripts/create_template.py`
5. Output: `introduction_reference.md` template

### Phase 2: Introduction Generation
1. Define research project and objectives
2. Collect 20-30 reference papers
3. Run generation: `python scripts/generate_intro.py`
4. Quality validation and accuracy scoring
5. Output: Draft introduction with integrated citations

## Technical Stack
- **PDF Processing**: PyMuPDF, pdfplumber
- **NLP**: spaCy, BioBERT, sentence-transformers
- **Vector DB**: Chroma
- **LLM**: OpenAI GPT-4 / Anthropic Claude
- **Citation**: bibtexparser, Zotero API
- **Backend**: FastAPI, LangChain
- **Frontend**: Streamlit
- **Storage**: SQLite for metadata

## Quality Control Features
- Cross-reference validation against source papers
- Citation context accuracy checking
- Confidence scoring for generated content
- Fact-checking mechanisms
- Style consistency validation

## Integration Features
- Zotero .bib file import/export
- Reference manager compatibility
- GitHub version control
- Automated quality reports
- Template customization

## Pattern Detection: Current Limitations and Improvements

### **Current Pattern Detection Reality:**
The existing pattern detection system is **terminology-focused and naive**, with significant limitations:

- **Terminology-Heavy Approach**: ~70% of pattern detection relies on neuroscience term density rather than sophisticated structural analysis
- **Basic Sentence Classification**: Uses simple regex patterns that miss nuanced argumentation structures
- **Shallow Structural Analysis**: "Funnel structure" detection only looks for keyword patterns, not actual conceptual flow
- **No Empirical Basis**: Current patterns are rule-based assumptions, not derived from analysis of actual published papers

### **Critical Gap Identified:**
The system fails to detect sophisticated writing patterns that distinguish high-quality introductions:
- **Conceptual Flow Progression**: How ideas develop from broad to specific across paragraphs
- **Argumentation Sophistication**: Complex reasoning structures and evidence integration
- **Transition Strategy Analysis**: How paragraphs connect conceptually and logically
- **Information Density Patterns**: Pacing and distribution of concepts throughout introductions

### **Planned Enhancements (Empirical Data Required):**

#### **Phase 1: Enhanced Universal Pattern Detection**
- **True Structural Analysis**: Detect conceptual breadth progression and logical flow patterns
- **Argumentation Mapping**: Identify problem→gap→solution vs. hypothesis→test→implications structures  
- **Transition Sophistication**: Analyze how paragraphs connect (causal, comparative, progressive)
- **Information Architecture**: Track paragraph function, length distribution, and concept density

#### **Phase 2: Empirical Pattern Learning**
- **Data Collection Framework**: Systematic analysis of 100+ successful introductions per research domain
- **Quantitative Structure Analysis**: Statistical measurement of paragraph counts, sentence complexity, argument flow
- **Success Pattern Identification**: Machine learning from published papers to identify effective structures
- **Validation Pipeline**: Cross-reference patterns against publication success metrics

#### **Future Journal-Specific Features (DATA-DEPENDENT)**
⚠️ **CRITICAL REQUIREMENT**: Any journal-specific patterns MUST be derived from empirical analysis of actual published papers
- **No Assumptions**: We make NO claims about journal preferences without statistical evidence
- **Empirical Validation Required**: 50+ paper analysis per journal before claiming structural differences  
- **Evidence-Based Only**: All recommendations must be backed by quantitative analysis of successful publications

### **Scientific Rigor Standards:**
- **Zero Unfounded Claims**: No assumptions about journal preferences or structural requirements
- **Empirical Evidence Requirement**: All patterns must be derived from actual paper analysis
- **Statistical Validation**: Quantitative proof required for any claimed structural differences
- **Honest Limitation Disclosure**: Clear communication about current system capabilities

## Development Notes
- **Pattern Detection Priority**: Focus on universal structural sophistication before any journal-specific features
- Implement robust error handling for PDF processing
- Maintain high accuracy standards for medical/scientific content
- Ensure proper citation attribution and context
- Build modular components for easy extension to other fields
- **Empirical Data Foundation**: All enhancements must be based on analysis of actual published papers

## Future Enhancements
- **Enhanced Pattern Detection**: Sophisticated structural analysis based on empirical paper analysis
- Multi-language support
- Additional citation styles
- Collaborative editing features
- Integration with manuscript preparation tools
- **Journal-Specific Features**: Only if supported by statistical analysis of published papers