"""PDF extraction and processing using multiple engines for robustness."""

import re
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

import fitz  # PyMuPDF
import pdfplumber

from .models import (
    ProcessedDocument, DocumentSection, Citation, DocumentMetadata,
    ProcessingStats, SectionType
)
from .exceptions import (
    PDFProcessingError, UnsupportedPDFError, ExtractionError,
    StructureDetectionError, CitationExtractionError
)

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Robust PDF extraction using multiple engines with fallback strategies."""
    
    def __init__(self):
        """Initialize the PDF extractor with configuration."""
        self.neuroscience_keywords = {
            'brain', 'neural', 'neuron', 'synapse', 'synaptic', 'cortex', 'hippocampus',
            'amygdala', 'cerebellum', 'thalamus', 'striatum', 'dopamine',
            'serotonin', 'glutamate', 'gaba', 'acetylcholine', 'neuroplasticity',
            'neurogenesis', 'neurodegeneration', 'cognitive', 'memory',
            'learning', 'attention', 'consciousness', 'epilepsy', 'seizure',
            'tumor', 'glioma', 'meningioma', 'neurosurgery', 'craniotomy',
            'functional', 'structural', 'connectivity', 'fmri', 'eeg', 'meg'
        }
        
        self.section_patterns = {
            SectionType.ABSTRACT: [
                r'abstract\b', r'summary\b'
            ],
            SectionType.INTRODUCTION: [
                r'introduction\b', r'background\b', r'overview\b'
            ],
            SectionType.METHODS: [
                r'methods?\b', r'methodology\b', r'materials?\b',
                r'experimental\s+procedures?\b', r'subjects?\b',
                r'participants?\b'
            ],
            SectionType.RESULTS: [
                r'results?\b', r'findings?\b', r'outcomes?\b'
            ],
            SectionType.DISCUSSION: [
                r'discussion\b', r'interpretation\b'
            ],
            SectionType.CONCLUSION: [
                r'conclusions?\b', r'summary\b', r'implications?\b'
            ],
            SectionType.REFERENCES: [
                r'references?\b', r'bibliography\b', r'citations?\b'
            ]
        }
    
    def process_pdf(self, file_path: str, use_fallback: bool = True) -> ProcessedDocument:
        """
        Process a PDF file and extract structured information.
        
        Args:
            file_path: Path to the PDF file
            use_fallback: Whether to use fallback extraction methods
            
        Returns:
            ProcessedDocument with extracted information
            
        Raises:
            PDFProcessingError: If processing fails
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise PDFProcessingError(f"File not found: {file_path}")
        
        if not file_path.suffix.lower() == '.pdf':
            raise UnsupportedPDFError(f"Not a PDF file: {file_path}")
        
        logger.info(f"Processing PDF: {file_path}")
        
        try:
            # Try primary extraction method (PyMuPDF)
            try:
                document_data = self._extract_with_pymupdf(file_path)
                extraction_method = "PyMuPDF"
            except Exception as e:
                logger.warning(f"PyMuPDF extraction failed: {e}")
                if not use_fallback:
                    raise ExtractionError(f"Primary extraction failed: {e}", str(file_path))
                
                # Try fallback method (pdfplumber)
                try:
                    document_data = self._extract_with_pdfplumber(file_path)
                    extraction_method = "pdfplumber"
                except Exception as e2:
                    logger.error(f"All extraction methods failed. PyMuPDF: {e}, pdfplumber: {e2}")
                    raise ExtractionError(f"All extraction methods failed: {e2}", str(file_path))
            
            # Process extracted data
            sections = self._detect_sections(document_data['text'], document_data['pages'])
            citations = self._extract_citations(document_data['text'])
            metadata = self._extract_metadata(document_data)
            
            # Assign citations to sections
            sections = self._assign_citations_to_sections(sections, citations)
            
            # Calculate statistics
            processing_time = time.time() - start_time
            stats = ProcessingStats(
                total_pages=document_data['page_count'],
                total_words=len(document_data['text'].split()),
                total_citations=len(citations),
                sections_detected=len(sections),
                processing_time=processing_time,
                extraction_method=extraction_method,
                confidence_score=self._calculate_confidence_score(sections, metadata),
                warnings=document_data.get('warnings', [])
            )
            
            return ProcessedDocument(
                file_path=str(file_path),
                metadata=metadata,
                sections=sections,
                full_text=document_data['text'],
                stats=stats,
                processed_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"PDF processing failed for {file_path}: {e}")
            if isinstance(e, PDFProcessingError):
                raise
            raise PDFProcessingError(f"Unexpected error: {e}", str(file_path))
    
    def _extract_with_pymupdf(self, file_path: Path) -> Dict[str, Any]:
        """Extract text using PyMuPDF with error handling."""
        try:
            doc = fitz.open(file_path)
            text_blocks = []
            pages_info = []
            warnings = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text blocks with position info
                blocks = page.get_text("dict")
                page_text = ""
                
                for block in blocks["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                page_text += span["text"] + " "
                
                if not page_text.strip():
                    warnings.append(f"Page {page_num + 1} appears to be empty or image-only")
                
                pages_info.append({
                    'page_num': page_num + 1,
                    'text': page_text.strip(),
                    'char_count': len(page_text)
                })
                text_blocks.append(page_text)
            
            doc.close()
            
            full_text = "\n\n".join(text_blocks)
            
            if not full_text.strip():
                raise ExtractionError("No text could be extracted from the PDF")
            
            return {
                'text': full_text,
                'pages': pages_info,
                'page_count': len(doc),
                'warnings': warnings
            }
            
        except Exception as e:
            raise ExtractionError(f"PyMuPDF extraction failed: {e}")
    
    def _extract_with_pdfplumber(self, file_path: Path) -> Dict[str, Any]:
        """Extract text using pdfplumber as fallback."""
        try:
            text_blocks = []
            pages_info = []
            warnings = []
            
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    
                    if not page_text:
                        warnings.append(f"Page {page_num + 1} could not be processed")
                        page_text = ""
                    
                    pages_info.append({
                        'page_num': page_num + 1,
                        'text': page_text,
                        'char_count': len(page_text) if page_text else 0
                    })
                    text_blocks.append(page_text or "")
                
                full_text = "\n\n".join(text_blocks)
                
                if not full_text.strip():
                    raise ExtractionError("No text could be extracted from the PDF")
                
                return {
                    'text': full_text,
                    'pages': pages_info,
                    'page_count': len(pdf.pages),
                    'warnings': warnings
                }
                
        except Exception as e:
            raise ExtractionError(f"pdfplumber extraction failed: {e}")
    
    def _detect_sections(self, text: str, pages_info: List[Dict]) -> List[DocumentSection]:
        """Detect document sections using pattern matching and structure analysis."""
        sections = []
        current_page = 1
        
        # Split text into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Track current section
        current_section_type = SectionType.UNKNOWN
        current_content = []
        current_title = ""
        section_start_page = 1
        
        for paragraph in paragraphs:
            # Check if this paragraph is a section header
            detected_type = self._identify_section_type(paragraph)
            
            if detected_type != SectionType.UNKNOWN:
                # Save previous section if it has content
                if current_content and current_section_type != SectionType.UNKNOWN:
                    content_text = "\n\n".join(current_content)
                    sections.append(DocumentSection(
                        section_type=current_section_type,
                        title=current_title,
                        content=content_text,
                        page_start=section_start_page,
                        page_end=self._estimate_page_number(content_text, pages_info),
                        citations=[],  # Will be filled later
                        word_count=len(content_text.split()),
                        confidence=0.8  # Base confidence for pattern-matched sections
                    ))
                
                # Start new section
                current_section_type = detected_type
                current_title = paragraph.strip()
                current_content = []
                section_start_page = self._estimate_page_number(paragraph, pages_info)
            else:
                # Add to current section content
                current_content.append(paragraph)
        
        # Add final section
        if current_content:
            content_text = "\n\n".join(current_content)
            sections.append(DocumentSection(
                section_type=current_section_type,
                title=current_title or "Content",
                content=content_text,
                page_start=section_start_page,
                page_end=self._estimate_page_number(content_text, pages_info),
                citations=[],
                word_count=len(content_text.split()),
                confidence=0.6  # Lower confidence for content without clear headers
            ))
        
        # If no sections detected, create one large section
        if not sections:
            sections.append(DocumentSection(
                section_type=SectionType.UNKNOWN,
                title="Full Document",
                content=text,
                page_start=1,
                page_end=len(pages_info),
                citations=[],
                word_count=len(text.split()),
                confidence=0.3
            ))
        
        return sections
    
    def _identify_section_type(self, text: str) -> SectionType:
        """Identify section type from text using patterns."""
        text_lower = text.lower().strip()
        
        # Skip if too long to be a header
        if len(text.split()) > 10:
            return SectionType.UNKNOWN
        
        # Check against patterns
        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return section_type
        
        return SectionType.UNKNOWN
    
    def _estimate_page_number(self, text: str, pages_info: List[Dict]) -> int:
        """Estimate page number for a given text snippet."""
        char_position = 0
        cumulative_chars = 0
        
        for page_info in pages_info:
            cumulative_chars += page_info['char_count']
            if char_position < cumulative_chars:
                return page_info['page_num']
        
        return len(pages_info)
    
    def _extract_citations(self, text: str) -> List[Citation]:
        """Extract citations using multiple pattern matching strategies."""
        citations = []
        
        # Pattern 1: Author (Year) format - more flexible
        pattern1 = r'([A-Z][a-zA-Z]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-zA-Z]+|[A-Z][a-zA-Z]+))*)\s*\((\d{4}[a-z]?)\)'
        matches1 = re.finditer(pattern1, text)
        
        for match in matches1:
            author_text = match.group(1).strip()
            year_text = match.group(2)
            year = int(re.search(r'\d{4}', year_text).group())
            
            # Parse authors
            if 'et al' in author_text:
                authors = [author_text.replace('et al.', '').replace('et al', '').strip()]
            elif '&' in author_text:
                authors = [a.strip() for a in author_text.split('&')]
            else:
                authors = [author_text]
            
            citations.append(Citation(
                text=match.group(0),
                authors=authors,
                year=year,
                confidence=0.8
            ))
        
        # Pattern 2: Multiple authors with commas (Smith, Jones, & Brown, 2020)
        pattern2 = r'([A-Z][a-zA-Z]+(?:,\s*[A-Z][a-zA-Z]+)*,?\s*&\s*[A-Z][a-zA-Z]+)\s*\((\d{4})\)'
        matches2 = re.finditer(pattern2, text)
        
        for match in matches2:
            author_text = match.group(1)
            year = int(match.group(2))
            
            # Parse comma-separated authors
            authors = [a.strip() for a in re.split(r'[,&]', author_text) if a.strip()]
            
            citations.append(Citation(
                text=match.group(0),
                authors=authors,
                year=year,
                confidence=0.9
            ))
        
        # Pattern 3: Simple author name patterns in references section
        ref_pattern = r'([A-Z][a-zA-Z]+,?\s+[A-Z]\.(?:\s*[A-Z]\.)?(?:,\s*[A-Z][a-zA-Z]+,?\s+[A-Z]\.(?:\s*[A-Z]\.)?)*)\s*\((\d{4})\)'
        ref_matches = re.finditer(ref_pattern, text)
        
        for match in ref_matches:
            authors_text = match.group(1)
            year = int(match.group(2))
            
            # Extract author surnames
            author_parts = re.findall(r'([A-Z][a-zA-Z]+),?\s+[A-Z]\.', authors_text)
            authors = [author.rstrip(',') for author in author_parts]
            
            citations.append(Citation(
                text=match.group(0),
                authors=authors,
                year=year,
                confidence=0.7
            ))
        
        # Remove duplicates based on similar text and year
        unique_citations = []
        seen_combinations = set()
        
        for citation in citations:
            # Create a normalized key for deduplication
            key = (citation.authors[0] if citation.authors else '', citation.year)
            if key not in seen_combinations and citation.authors:
                unique_citations.append(citation)
                seen_combinations.add(key)
        
        return unique_citations
    
    def _extract_metadata(self, document_data: Dict[str, Any]) -> DocumentMetadata:
        """Extract document metadata from text and structure."""
        text = document_data['text']
        
        # Try to extract title (usually first few lines)
        lines = text.split('\n')[:10]
        potential_title = None
        
        for line in lines:
            line = line.strip()
            if len(line) > 20 and not line.isupper() and '.' not in line[:50]:
                potential_title = line
                break
        
        # Extract DOI
        doi_pattern = r'doi[:\s]*([0-9]{2}\.[0-9]{4}/[^\s]+)'
        doi_match = re.search(doi_pattern, text, re.IGNORECASE)
        doi = doi_match.group(1) if doi_match else None
        
        # Extract PMID
        pmid_pattern = r'pmid[:\s]*([0-9]+)'
        pmid_match = re.search(pmid_pattern, text, re.IGNORECASE)
        pmid = pmid_match.group(1) if pmid_match else None
        
        # Extract keywords related to neuroscience
        found_keywords = []
        text_lower = text.lower()
        for keyword in self.neuroscience_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return DocumentMetadata(
            title=potential_title,
            authors=[],  # Complex extraction - could be enhanced
            doi=doi,
            pmid=pmid,
            keywords=found_keywords[:10]  # Limit to top 10
        )
    
    def _assign_citations_to_sections(self, sections: List[DocumentSection], 
                                    citations: List[Citation]) -> List[DocumentSection]:
        """Assign citations to their respective sections."""
        for section in sections:
            section_citations = []
            for citation in citations:
                if citation.text in section.content:
                    section_citations.append(citation)
            section.citations = section_citations
        
        return sections
    
    def _calculate_confidence_score(self, sections: List[DocumentSection], 
                                  metadata: DocumentMetadata) -> float:
        """Calculate overall confidence score for the extraction."""
        scores = []
        
        # Section detection confidence
        section_types = [s.section_type for s in sections]
        if SectionType.INTRODUCTION in section_types:
            scores.append(0.3)
        if SectionType.ABSTRACT in section_types:
            scores.append(0.2)
        if SectionType.METHODS in section_types:
            scores.append(0.2)
        if SectionType.RESULTS in section_types:
            scores.append(0.2)
        
        # Metadata confidence
        if metadata.title:
            scores.append(0.1)
        if metadata.doi or metadata.pmid:
            scores.append(0.1)
        if metadata.keywords:
            scores.append(0.1)
        
        return min(sum(scores), 1.0)