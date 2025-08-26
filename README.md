# Neuro-Scientific Writing Assistant 🧠✍️

An AI-powered scientific writing companion specialized in neuroscience/neurosurgery that learns from reference papers to generate structured introduction templates and drafts.

## 🎯 Purpose

This tool helps neuroscience researchers:
- **Learn writing patterns** from published neuroscience papers
- **Generate templates** based on field-specific conventions
- **Create introduction drafts** with properly integrated citations
- **Ensure quality** through automated validation against source papers

## 🚀 Quick Start

1. **Setup Environment**
   ```bash
   git clone <repository-url>
   cd neuro-scientific-writer
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Train Template**
   ```bash
   # Add PDF papers to data/training_papers/
   python scripts/train_template.py --input data/training_papers/
   ```

3. **Generate Introduction**
   ```bash
   python scripts/generate_intro.py --project "Your Research Title" --refs data/references/
   ```

## 📋 Features

- ✅ **PDF Processing**: Extract and analyze scientific papers
- ✅ **Style Learning**: Understand neuroscience writing patterns  
- ✅ **Template Generation**: Create field-specific introduction frameworks
- ✅ **Citation Integration**: Smart reference weaving with context
- ✅ **Quality Control**: Accuracy validation against source papers
- ✅ **Zotero Integration**: Compatible with reference managers
- ✅ **Web Interface**: User-friendly Streamlit app

## 🏗️ Architecture

```
src/
├── pdf_processor/     # PDF extraction and parsing
├── analysis/          # NLP and content analysis  
├── template_engine/   # Template generation
├── citation_manager/  # Reference handling
├── quality_control/   # Validation and fact-checking
└── api/              # FastAPI backend
```

## 📖 Workflow

1. **Training Phase**: Analyze 20-50 neuroscience papers → generate domain template
2. **Generation Phase**: Input project + 20-30 references → produce introduction draft  
3. **Validation Phase**: Quality check against source papers with accuracy metrics

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI, LangChain
- **NLP**: spaCy, BioBERT, sentence-transformers
- **Storage**: Chroma vector DB, SQLite
- **Frontend**: Streamlit
- **Citation**: bibtexparser, Zotero API

## 🎓 Target Fields

Primary focus: Neuroscience, Neurosurgery, Neuro-oncology
Citation style: APA (flexible output)

## 📚 Documentation

See `CLAUDE.md` for detailed development context and commands.

## 🤝 Contributing

This is a specialized research tool. Contributions welcome for:
- Additional citation styles
- Field-specific extensions  
- Quality control improvements
- Integration enhancements

## 📄 License

MIT License - See LICENSE file for details.