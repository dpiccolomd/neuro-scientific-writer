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

## Development Notes
- Focus on neuroscience-specific terminology and writing patterns
- Implement robust error handling for PDF processing
- Maintain high accuracy standards for medical/scientific content
- Ensure proper citation attribution and context
- Build modular components for easy extension to other fields

## Future Enhancements
- Multi-language support
- Additional citation styles
- Collaborative editing features
- Integration with manuscript preparation tools
- Advanced statistical analysis of writing patterns