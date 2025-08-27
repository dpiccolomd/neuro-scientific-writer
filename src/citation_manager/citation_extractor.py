"""Citation extraction from text documents."""

import re
import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

from .models import Citation, Reference, CitationContext
from .exceptions import CitationExtractionError

logger = logging.getLogger(__name__)


class CitationExtractor:
    """Extract citations from scientific text."""
    
    def __init__(self):
        """Initialize citation extractor with pattern recognition."""
        self.citation_patterns = self._compile_citation_patterns()
        self.reference_patterns = self._compile_reference_patterns()
        
    def extract_citations(self, text: str, document_id: str = None) -> List[Citation]:
        """
        Extract in-text citations from document text.
        
        Args:
            text: Document text to analyze
            document_id: Optional document identifier
            
        Returns:
            List of extracted Citation objects
            
        Raises:
            CitationExtractionError: If extraction fails
        """
        if not text or not text.strip():
            return []
        
        try:
            citations = []
            citation_id_counter = 1
            
            # Find all citation matches
            for pattern_name, pattern in self.citation_patterns.items():
                matches = list(pattern.finditer(text))
                
                for match in matches:
                    citation = self._create_citation_from_match(
                        match, pattern_name, citation_id_counter, document_id
                    )
                    if citation:
                        citations.append(citation)
                        citation_id_counter += 1
            
            # Remove duplicates based on position and reference
            citations = self._deduplicate_citations(citations)
            
            # Sort by position in text
            citations.sort(key=lambda c: c.position_in_text)
            
            logger.info(f"Extracted {len(citations)} citations from text")
            return citations
            
        except Exception as e:
            logger.error(f"Citation extraction failed: {e}")
            raise CitationExtractionError(f"Failed to extract citations: {e}")
    
    def extract_references(self, text: str) -> List[Dict]:
        """
        Extract reference list from bibliography section.
        
        Args:
            text: Bibliography text
            
        Returns:
            List of reference dictionaries with extracted information
        """
        try:
            references = []
            
            # Find reference section
            ref_section = self._find_reference_section(text)
            if not ref_section:
                logger.warning("No reference section found")
                return []
            
            # Split into individual references
            ref_entries = self._split_reference_entries(ref_section)
            
            for i, entry in enumerate(ref_entries):
                ref_data = self._parse_reference_entry(entry, i + 1)
                if ref_data:
                    references.append(ref_data)
            
            logger.info(f"Extracted {len(references)} references from bibliography")
            return references
            
        except Exception as e:
            logger.error(f"Reference extraction failed: {e}")
            raise CitationExtractionError(f"Failed to extract references: {e}")
    
    def extract_citation_contexts(self, text: str, citations: List[Citation]) -> List[CitationContext]:
        """
        Extract context information for citations.
        
        Args:
            text: Full document text
            citations: List of citations to analyze
            
        Returns:
            List of CitationContext objects
        """
        try:
            contexts = []
            sentences = self._split_into_sentences(text)
            paragraphs = self._split_into_paragraphs(text)
            
            for citation in citations:
                context = self._extract_citation_context(
                    citation, text, sentences, paragraphs
                )
                if context:
                    contexts.append(context)
            
            logger.info(f"Extracted contexts for {len(contexts)} citations")
            return contexts
            
        except Exception as e:
            logger.error(f"Context extraction failed: {e}")
            raise CitationExtractionError(f"Failed to extract citation contexts: {e}")
    
    def _compile_citation_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regular expressions for citation patterns."""
        patterns = {}
        
        # Parenthetical citations: (Author, Year), (Author et al., Year)
        patterns['parenthetical'] = re.compile(
            r'\(([A-Za-z\-\']+(?:\s+(?:et\s+al\.?|&\s+[A-Za-z\-\']+))?),?\s*(\d{4}[a-z]?)(?:,\s*p\.?\s*(\d+(?:-\d+)?))?(?:;\s*[^)]+)?\)',
            re.IGNORECASE
        )
        
        # Narrative citations: Author (Year), Author et al. (Year)
        patterns['narrative'] = re.compile(
            r'([A-Za-z\-\']+(?:\s+(?:et\s+al\.?|&\s+[A-Za-z\-\']+))?)\s+\((\d{4}[a-z]?)(?:,\s*p\.?\s*(\d+(?:-\d+)?))?\)',
            re.IGNORECASE
        )
        
        # Multiple citations: (Author1, Year1; Author2, Year2)
        patterns['multiple'] = re.compile(
            r'\(([^)]+;\s*[^)]+)\)',
            re.IGNORECASE
        )
        
        # Numbered citations: [1], [1-3], [1,2,3]
        patterns['numbered'] = re.compile(
            r'\[(\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*)\]'
        )
        
        return patterns
    
    def _compile_reference_patterns(self) -> Dict[str, re.Pattern]:
        """Compile patterns for reference parsing."""
        patterns = {}
        
        # Journal article pattern
        patterns['journal'] = re.compile(
            r'^(.+?)\s+\((\d{4})\)\.\s*(.+?)\.\s*(.+?),?\s*(\d+(?:\(\d+\))?)(?:,\s*(\d+(?:-\d+)?))?\.'
        )
        
        # Book pattern  
        patterns['book'] = re.compile(
            r'^(.+?)\s+\((\d{4})\)\.\s*(.+?)\.\s*([^.]+):\s*([^.]+)\.'
        )
        
        # DOI pattern
        patterns['doi'] = re.compile(
            r'doi:?\s*(10\.\d+/[^\s]+)|https?://doi\.org/(10\.\d+/[^\s]+)',
            re.IGNORECASE
        )
        
        return patterns
    
    def _create_citation_from_match(
        self, 
        match: re.Match, 
        pattern_type: str, 
        citation_id: int,
        document_id: str = None
    ) -> Optional[Citation]:
        """Create Citation object from regex match."""
        try:
            # Extract position and context
            start_pos = match.start()
            end_pos = match.end()
            full_match = match.group(0)
            
            # Generate citation ID
            doc_prefix = f"{document_id}_" if document_id else ""
            citation_id_str = f"{doc_prefix}citation_{citation_id}"
            
            # Parse based on pattern type
            if pattern_type == 'parenthetical':
                author = match.group(1)
                year = match.group(2)
                page = match.group(3) if match.lastindex >= 3 else None
                
                return Citation(
                    citation_id=citation_id_str,
                    reference_id=f"ref_{author.lower().replace(' ', '_')}_{year}",
                    page_number=page,
                    is_parenthetical=True,
                    position_in_text=start_pos
                )
                
            elif pattern_type == 'narrative':
                author = match.group(1)
                year = match.group(2)
                page = match.group(3) if match.lastindex >= 3 else None
                
                return Citation(
                    citation_id=citation_id_str,
                    reference_id=f"ref_{author.lower().replace(' ', '_')}_{year}",
                    page_number=page,
                    is_parenthetical=False,
                    position_in_text=start_pos
                )
                
            elif pattern_type == 'numbered':
                numbers = match.group(1)
                # For now, use the first number as reference
                ref_num = numbers.split(',')[0].split('-')[0]
                
                return Citation(
                    citation_id=citation_id_str,
                    reference_id=f"ref_num_{ref_num}",
                    is_parenthetical=True,
                    position_in_text=start_pos
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to create citation from match: {e}")
            return None
    
    def _deduplicate_citations(self, citations: List[Citation]) -> List[Citation]:
        """Remove duplicate citations based on position and reference."""
        seen = set()
        unique_citations = []
        
        for citation in citations:
            key = (citation.reference_id, citation.position_in_text)
            if key not in seen:
                seen.add(key)
                unique_citations.append(citation)
        
        return unique_citations
    
    def _find_reference_section(self, text: str) -> Optional[str]:
        """Find and extract reference section from text."""
        # Common reference section headers
        ref_headers = [
            r'references?\s*$',
            r'bibliography\s*$',
            r'works?\s+cited\s*$',
            r'literature\s+cited\s*$'
        ]
        
        text_lines = text.split('\n')
        ref_start_idx = None
        
        for i, line in enumerate(text_lines):
            line_lower = line.lower().strip()
            for pattern in ref_headers:
                if re.match(pattern, line_lower):
                    ref_start_idx = i
                    break
            if ref_start_idx is not None:
                break
        
        if ref_start_idx is None:
            return None
        
        # Extract from reference header to end of document
        return '\n'.join(text_lines[ref_start_idx + 1:])
    
    def _split_reference_entries(self, ref_text: str) -> List[str]:
        """Split reference section into individual entries."""
        # Split by common patterns that indicate new references
        entries = []
        
        # Try numbered references first
        numbered_pattern = r'^\s*\d+\.\s+'
        if re.search(numbered_pattern, ref_text, re.MULTILINE):
            entries = re.split(numbered_pattern, ref_text, flags=re.MULTILINE)
            entries = [e.strip() for e in entries if e.strip()]
        else:
            # Fall back to paragraph-based splitting
            paragraphs = ref_text.split('\n\n')
            entries = [p.strip() for p in paragraphs if p.strip()]
        
        return entries
    
    def _parse_reference_entry(self, entry: str, ref_num: int) -> Optional[Dict]:
        """Parse individual reference entry."""
        try:
            # Clean up the entry
            entry = re.sub(r'\s+', ' ', entry.strip())
            
            # Try journal pattern first
            journal_match = self.reference_patterns['journal'].match(entry)
            if journal_match:
                return {
                    'reference_id': f"ref_{ref_num}",
                    'authors': journal_match.group(1),
                    'year': int(journal_match.group(2)),
                    'title': journal_match.group(3),
                    'journal': journal_match.group(4),
                    'volume': journal_match.group(5),
                    'pages': journal_match.group(6),
                    'type': 'journal_article',
                    'raw_text': entry
                }
            
            # Try book pattern
            book_match = self.reference_patterns['book'].match(entry)
            if book_match:
                return {
                    'reference_id': f"ref_{ref_num}",
                    'authors': book_match.group(1),
                    'year': int(book_match.group(2)),
                    'title': book_match.group(3),
                    'place': book_match.group(4),
                    'publisher': book_match.group(5),
                    'type': 'book',
                    'raw_text': entry
                }
            
            # Extract DOI if present
            doi_match = self.reference_patterns['doi'].search(entry)
            doi = None
            if doi_match:
                doi = doi_match.group(1) or doi_match.group(2)
            
            # Return partial information
            return {
                'reference_id': f"ref_{ref_num}",
                'raw_text': entry,
                'doi': doi,
                'type': 'unknown'
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse reference entry: {e}")
            return None
    
    def _extract_citation_context(
        self, 
        citation: Citation, 
        full_text: str, 
        sentences: List[str], 
        paragraphs: List[str]
    ) -> Optional[CitationContext]:
        """Extract context information for a citation."""
        try:
            # Find the sentence containing the citation
            containing_sentence = ""
            for sentence in sentences:
                if citation.position_in_text >= full_text.find(sentence) and \
                   citation.position_in_text < full_text.find(sentence) + len(sentence):
                    containing_sentence = sentence
                    break
            
            # Find the paragraph containing the citation
            containing_paragraph = ""
            for paragraph in paragraphs:
                if citation.position_in_text >= full_text.find(paragraph) and \
                   citation.position_in_text < full_text.find(paragraph) + len(paragraph):
                    containing_paragraph = paragraph
                    break
            
            # Determine semantic role (simplified)
            semantic_role = self._determine_semantic_role(containing_sentence)
            
            return CitationContext(
                citation=citation,
                sentence=containing_sentence,
                paragraph=containing_paragraph,
                section="unknown",  # Would need section detection
                semantic_role=semantic_role,
                confidence_score=0.8  # Default confidence
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract context for citation {citation.citation_id}: {e}")
            return None
    
    def _determine_semantic_role(self, sentence: str) -> str:
        """Determine the semantic role of a citation in context."""
        sentence_lower = sentence.lower()
        
        # Supporting evidence
        if any(word in sentence_lower for word in ['showed', 'demonstrated', 'found', 'reported']):
            return 'supporting'
        
        # Contrasting evidence
        if any(word in sentence_lower for word in ['however', 'but', 'contrary', 'unlike']):
            return 'contrasting'
        
        # Methodological
        if any(word in sentence_lower for word in ['method', 'technique', 'approach', 'protocol']):
            return 'methodological'
        
        # Background
        if any(word in sentence_lower for word in ['previous', 'prior', 'earlier', 'established']):
            return 'background'
        
        return 'general'
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - could be improved with NLP libraries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]