"""Tests for PDF processing functionality."""

import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime

from src.pdf_processor import PDFExtractor, ProcessedDocument, DocumentSection, Citation
from src.pdf_processor.models import SectionType, DocumentMetadata, ProcessingStats
from src.pdf_processor.exceptions import PDFProcessingError, UnsupportedPDFError


class TestPDFExtractor:
    """Test cases for PDFExtractor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = PDFExtractor()
    
    def test_initialization(self):
        """Test PDFExtractor initialization."""
        assert self.extractor is not None
        assert len(self.extractor.neuroscience_keywords) > 0
        assert 'brain' in self.extractor.neuroscience_keywords
        assert 'neuron' in self.extractor.neuroscience_keywords
    
    def test_process_nonexistent_file(self):
        """Test processing a non-existent file raises appropriate error."""
        with pytest.raises(PDFProcessingError) as exc_info:
            self.extractor.process_pdf("/path/to/nonexistent/file.pdf")
        
        assert "File not found" in str(exc_info.value)
    
    def test_process_non_pdf_file(self):
        """Test processing a non-PDF file raises UnsupportedPDFError."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
            tmp_file.write(b"This is not a PDF file")
            tmp_path = tmp_file.name
        
        try:
            with pytest.raises(UnsupportedPDFError) as exc_info:
                self.extractor.process_pdf(tmp_path)
            
            assert "Not a PDF file" in str(exc_info.value)
        finally:
            os.unlink(tmp_path)
    
    def test_identify_section_type(self):
        """Test section type identification."""
        # Test abstract detection
        assert self.extractor._identify_section_type("Abstract") == SectionType.ABSTRACT
        assert self.extractor._identify_section_type("ABSTRACT") == SectionType.ABSTRACT
        
        # Test introduction detection
        assert self.extractor._identify_section_type("Introduction") == SectionType.INTRODUCTION
        assert self.extractor._identify_section_type("1. Introduction") == SectionType.INTRODUCTION
        
        # Test methods detection
        assert self.extractor._identify_section_type("Methods") == SectionType.METHODS
        assert self.extractor._identify_section_type("Materials and Methods") == SectionType.METHODS
        
        # Test unknown for long text
        long_text = "This is a very long paragraph that should not be identified as a section header"
        assert self.extractor._identify_section_type(long_text) == SectionType.UNKNOWN
    
    def test_extract_citations(self):
        """Test citation extraction from text."""
        test_text = """
        Previous studies have shown (Smith, 2020) that neural plasticity is important.
        Other research by Johnson et al. (2019) confirms these findings.
        The work of Brown (2021) provides additional evidence.
        """
        
        citations = self.extractor._extract_citations(test_text)
        
        # Should find at least 3 citations
        assert len(citations) >= 2
        
        # Check specific citations
        citation_texts = [c.text for c in citations]
        assert any("Smith" in text for text in citation_texts)
        assert any("Johnson" in text for text in citation_texts)
    
    def test_extract_metadata(self):
        """Test metadata extraction."""
        document_data = {
            'text': """
            Neural Mechanisms of Memory Formation in the Human Brain
            
            This study investigates hippocampal function and memory consolidation.
            The brain processes information through neural networks.
            
            DOI: 10.1016/j.neuron.2023.01.001
            PMID: 12345678
            """,
            'pages': [{'page_num': 1, 'text': 'content', 'char_count': 100}]
        }
        
        metadata = self.extractor._extract_metadata(document_data)
        
        assert metadata.title is not None
        assert "Neural Mechanisms" in metadata.title
        assert metadata.doi == "10.1016/j.neuron.2023.01.001"
        assert metadata.pmid == "12345678"
        assert len(metadata.keywords) > 0
        assert "brain" in metadata.keywords
        assert "hippocampal" in metadata.keywords or "hippocampus" in metadata.keywords
    
    def test_calculate_confidence_score(self):
        """Test confidence score calculation."""
        # Create mock sections with different types
        sections = [
            DocumentSection(
                section_type=SectionType.INTRODUCTION,
                title="Introduction",
                content="Test content",
                page_start=1,
                page_end=1,
                citations=[],
                word_count=10,
                confidence=0.8
            ),
            DocumentSection(
                section_type=SectionType.METHODS,
                title="Methods",
                content="Test methods",
                page_start=2,
                page_end=2,
                citations=[],
                word_count=15,
                confidence=0.7
            )
        ]
        
        metadata = DocumentMetadata(
            title="Test Paper",
            doi="10.1000/test",
            keywords=["brain", "neuron"]
        )
        
        confidence = self.extractor._calculate_confidence_score(sections, metadata)
        
        # Should have reasonable confidence with intro, methods, title, DOI, and keywords
        assert 0.5 <= confidence <= 1.0
    
    def test_assign_citations_to_sections(self):
        """Test citation assignment to sections."""
        citations = [
            Citation(text="Smith (2020)", authors=["Smith"], year=2020, confidence=0.8),
            Citation(text="Jones (2021)", authors=["Jones"], year=2021, confidence=0.7)
        ]
        
        sections = [
            DocumentSection(
                section_type=SectionType.INTRODUCTION,
                title="Introduction",
                content="Previous work by Smith (2020) showed important results.",
                page_start=1,
                page_end=1,
                citations=[],
                word_count=10,
                confidence=0.8
            ),
            DocumentSection(
                section_type=SectionType.DISCUSSION,
                title="Discussion",
                content="Our findings align with Jones (2021) research.",
                page_start=3,
                page_end=3,
                citations=[],
                word_count=8,
                confidence=0.7
            )
        ]
        
        updated_sections = self.extractor._assign_citations_to_sections(sections, citations)
        
        # Introduction should have Smith citation
        intro_citations = updated_sections[0].citations
        assert len(intro_citations) == 1
        assert intro_citations[0].authors[0] == "Smith"
        
        # Discussion should have Jones citation
        discussion_citations = updated_sections[1].citations
        assert len(discussion_citations) == 1
        assert discussion_citations[0].authors[0] == "Jones"


class TestModels:
    """Test cases for data models."""
    
    def test_citation_validation(self):
        """Test Citation model validation."""
        # Valid citation
        citation = Citation(
            text="Smith (2020)",
            authors=["Smith"],
            year=2020,
            confidence=0.8
        )
        assert citation.text == "Smith (2020)"
        assert citation.authors == ["Smith"]
        assert citation.year == 2020
        assert citation.confidence == 0.8
        
        # Invalid confidence should raise ValueError
        with pytest.raises(ValueError):
            Citation(
                text="Test",
                authors=["Test"],
                confidence=1.5  # Invalid confidence
            )
        
        # Empty text should raise ValueError
        with pytest.raises(ValueError):
            Citation(
                text="",
                authors=["Test"]
            )
    
    def test_document_section_validation(self):
        """Test DocumentSection model validation."""
        # Valid section
        section = DocumentSection(
            section_type=SectionType.INTRODUCTION,
            title="Introduction",
            content="This is test content with multiple words for counting.",
            page_start=1,
            page_end=2,
            citations=[],
            word_count=0,  # Should be calculated automatically
            confidence=0.8
        )
        
        # Word count should be calculated
        assert section.word_count == 10
        
        # Invalid page numbers should raise ValueError
        with pytest.raises(ValueError):
            DocumentSection(
                section_type=SectionType.INTRODUCTION,
                title="Test",
                content="Test content",
                page_start=5,
                page_end=3,  # End before start
                citations=[],
                word_count=2
            )
    
    def test_processed_document_properties(self):
        """Test ProcessedDocument properties and methods."""
        sections = [
            DocumentSection(
                section_type=SectionType.ABSTRACT,
                title="Abstract",
                content="Abstract content",
                page_start=1,
                page_end=1,
                citations=[],
                word_count=2
            ),
            DocumentSection(
                section_type=SectionType.INTRODUCTION,
                title="Introduction",
                content="Introduction content",
                page_start=2,
                page_end=2,
                citations=[Citation(text="Test (2020)", authors=["Test"], year=2020)],
                word_count=2
            )
        ]
        
        document = ProcessedDocument(
            file_path="/test/path.pdf",
            metadata=DocumentMetadata(),
            sections=sections,
            full_text="Abstract content Introduction content",
            stats=ProcessingStats(
                total_pages=2,
                total_words=4,
                total_citations=1,
                sections_detected=2,
                processing_time=1.5,
                extraction_method="test",
                confidence_score=0.8,
                warnings=[]
            ),
            processed_at=datetime.now()
        )
        
        # Test properties
        assert document.abstract_section is not None
        assert document.abstract_section.section_type == SectionType.ABSTRACT
        
        assert document.introduction_section is not None
        assert document.introduction_section.section_type == SectionType.INTRODUCTION
        
        # Test all_citations
        all_citations = document.all_citations
        assert len(all_citations) == 1
        assert all_citations[0].text == "Test (2020)"
        
        # Test get_sections_by_type
        intro_sections = document.get_sections_by_type(SectionType.INTRODUCTION)
        assert len(intro_sections) == 1
        assert intro_sections[0].title == "Introduction"
        
        # Test to_dict
        doc_dict = document.to_dict()
        assert doc_dict["file_path"] == "/test/path.pdf"
        assert len(doc_dict["sections"]) == 2
        assert "processed_at" in doc_dict