"""APA citation formatting implementation."""

import re
import logging
from typing import List, Dict, Optional
from datetime import datetime

from .models import (
    Reference, Citation, Author, Journal, CitationType, 
    BibliographyEntry, CitationStyle
)
from .exceptions import APAFormattingError

logger = logging.getLogger(__name__)


class APAFormatter:
    """APA (7th edition) citation formatter for neuroscience literature."""
    
    def __init__(self):
        """Initialize APA formatter with neuroscience-specific rules."""
        self.style = CitationStyle.APA
        self.neuroscience_journals = self._load_neuroscience_journal_abbreviations()
        self.author_limit = 20  # Maximum authors before "et al."
        
    def format_reference(self, reference: Reference) -> BibliographyEntry:
        """
        Format a reference according to APA 7th edition style.
        
        Args:
            reference: Reference to format
            
        Returns:
            BibliographyEntry with formatted text
            
        Raises:
            APAFormattingError: If formatting fails
        """
        try:
            if reference.citation_type == CitationType.JOURNAL_ARTICLE:
                formatted = self._format_journal_article(reference)
            elif reference.citation_type == CitationType.BOOK:
                formatted = self._format_book(reference)
            elif reference.citation_type == CitationType.BOOK_CHAPTER:
                formatted = self._format_book_chapter(reference)
            elif reference.citation_type == CitationType.CONFERENCE_PAPER:
                formatted = self._format_conference_paper(reference)
            elif reference.citation_type == CitationType.PREPRINT:
                formatted = self._format_preprint(reference)
            else:
                raise APAFormattingError(f"Unsupported citation type: {reference.citation_type}")
            
            # Create sort key for bibliography ordering
            sort_key = self._generate_sort_key(reference)
            
            return BibliographyEntry(
                reference_id=reference.reference_id,
                formatted_text=formatted,
                style=self.style,
                sort_key=sort_key
            )
            
        except Exception as e:
            logger.error(f"Failed to format reference {reference.reference_id}: {e}")
            raise APAFormattingError(f"Reference formatting failed: {e}")
    
    def format_in_text_citation(self, citation: Citation, reference: Reference) -> str:
        """
        Format in-text citation according to APA style.
        
        Args:
            citation: Citation instance
            reference: Referenced work
            
        Returns:
            Formatted in-text citation
        """
        try:
            # Handle author names
            if len(reference.authors) == 1:
                author_part = reference.authors[0].last_name
            elif len(reference.authors) == 2:
                author_part = f"{reference.authors[0].last_name} & {reference.authors[1].last_name}"
            elif len(reference.authors) >= 3:
                author_part = f"{reference.authors[0].last_name} et al."
            else:
                author_part = "Anonymous"
            
            # Base citation
            base_citation = f"{author_part}, {reference.year}"
            
            # Add page numbers if provided
            if citation.page_number:
                base_citation += f", p. {citation.page_number}"
            
            # Handle prefix and suffix
            citation_text = base_citation
            if citation.prefix:
                citation_text = f"{citation.prefix} {citation_text}"
            if citation.suffix:
                citation_text = f"{citation_text}, {citation.suffix}"
            
            # Handle parenthetical vs narrative
            if citation.is_parenthetical:
                return f"({citation_text})"
            else:
                return citation_text
                
        except Exception as e:
            logger.error(f"Failed to format in-text citation: {e}")
            raise APAFormattingError(f"In-text citation formatting failed: {e}")
    
    def _format_journal_article(self, ref: Reference) -> str:
        """Format journal article according to APA style."""
        # Authors
        authors = self._format_authors(ref.authors)
        
        # Year
        year = f"({ref.year})"
        
        # Title (sentence case, no quotes)
        title = self._format_title(ref.title)
        
        # Journal name (italicized, title case)
        journal_name = self._format_journal_name(ref.journal.name if ref.journal else "Unknown Journal")
        
        # Volume and issue
        vol_issue = ""
        if ref.volume:
            vol_issue = f"*{ref.volume}*"
            if ref.issue:
                vol_issue += f"({ref.issue})"
        
        # Pages
        pages = ""
        if ref.pages:
            pages = self._format_pages(ref.pages)
        
        # DOI
        doi = ""
        if ref.doi:
            doi = f"https://doi.org/{ref.doi.replace('doi:', '').replace('DOI:', '')}"
        
        # Assemble citation
        parts = [authors, year, title, journal_name]
        
        if vol_issue:
            parts.append(vol_issue)
        if pages:
            parts.append(pages)
        if doi:
            parts.append(doi)
        
        # Join with appropriate punctuation
        citation = f"{authors} {year}. {title}. *{journal_name}*"
        if vol_issue:
            citation += f", {vol_issue}"
        if pages:
            citation += f", {pages}"
        if doi:
            citation += f". {doi}"
        else:
            citation += "."
            
        return citation
    
    def _format_book(self, ref: Reference) -> str:
        """Format book according to APA style."""
        authors = self._format_authors(ref.authors)
        year = f"({ref.year})"
        title = f"*{self._format_title(ref.title)}*"
        
        publisher_info = ""
        if ref.publisher:
            publisher_info = ref.publisher
            if ref.place:
                publisher_info = f"{ref.place}: {publisher_info}"
        
        citation = f"{authors} {year}. {title}."
        if publisher_info:
            citation += f" {publisher_info}."
            
        return citation
    
    def _format_book_chapter(self, ref: Reference) -> str:
        """Format book chapter according to APA style."""
        # This would need additional fields in the Reference model
        # for book editor, book title, etc.
        return self._format_book(ref)  # Simplified for now
    
    def _format_conference_paper(self, ref: Reference) -> str:
        """Format conference paper according to APA style."""
        return self._format_journal_article(ref)  # Simplified for now
    
    def _format_preprint(self, ref: Reference) -> str:
        """Format preprint according to APA style."""
        authors = self._format_authors(ref.authors)
        year = f"({ref.year})"
        title = self._format_title(ref.title)
        
        citation = f"{authors} {year}. {title}. *Preprint*"
        if ref.doi:
            citation += f". https://doi.org/{ref.doi.replace('doi:', '').replace('DOI:', '')}"
        elif ref.url:
            citation += f". {ref.url}"
        else:
            citation += "."
            
        return citation
    
    def _format_authors(self, authors: List[Author]) -> str:
        """Format author list according to APA style."""
        if not authors:
            return "Anonymous"
        
        if len(authors) == 1:
            return authors[0].apa_format
        elif len(authors) <= 20:
            formatted_authors = [author.apa_format for author in authors[:-1]]
            last_author = authors[-1].apa_format
            return f"{', '.join(formatted_authors)}, & {last_author}"
        else:
            # More than 20 authors - use first 19, then "...", then last
            first_authors = [author.apa_format for author in authors[:19]]
            last_author = authors[-1].apa_format
            return f"{', '.join(first_authors)}, ... {last_author}"
    
    def _format_title(self, title: str) -> str:
        """Format title in sentence case."""
        # Remove extra whitespace and ensure sentence case
        title = re.sub(r'\s+', ' ', title.strip())
        
        # Basic sentence case conversion (first word capitalized)
        if title:
            title = title[0].upper() + title[1:].lower()
        
        # Capitalize after colons and preserve proper nouns (simplified)
        title = re.sub(r':\s*([a-z])', lambda m: f': {m.group(1).upper()}', title)
        
        return title
    
    def _format_journal_name(self, journal_name: str) -> str:
        """Format journal name with proper abbreviation if available."""
        # Check if we have a standard abbreviation
        journal_lower = journal_name.lower()
        for full_name, abbrev in self.neuroscience_journals.items():
            if full_name.lower() == journal_lower:
                return abbrev
        
        # Return original if no abbreviation found
        return journal_name
    
    def _format_pages(self, pages: str) -> str:
        """Format page numbers according to APA style."""
        # Handle different page formats
        if '-' in pages or '–' in pages or '—' in pages:
            return pages.replace('--', '–').replace('—', '–')
        else:
            return pages
    
    def _generate_sort_key(self, reference: Reference) -> str:
        """Generate sort key for bibliography ordering."""
        # APA sorts by: Author surname, year, title
        first_author = reference.authors[0].last_name if reference.authors else "Anonymous"
        year_str = f"{reference.year:04d}"
        title_start = reference.title[:20].lower() if reference.title else ""
        
        return f"{first_author.lower()}_{year_str}_{title_start}"
    
    def _load_neuroscience_journal_abbreviations(self) -> Dict[str, str]:
        """Load standard neuroscience journal abbreviations."""
        return {
            "Nature Neuroscience": "Nat. Neurosci.",
            "Neuron": "Neuron",
            "Journal of Neuroscience": "J. Neurosci.",
            "Brain": "Brain",
            "Cerebral Cortex": "Cereb. Cortex",
            "NeuroImage": "NeuroImage",
            "Human Brain Mapping": "Hum. Brain Mapp.",
            "Journal of Neurophysiology": "J. Neurophysiol.",
            "Proceedings of the National Academy of Sciences": "PNAS",
            "Science": "Science",
            "Nature": "Nature",
            "Current Biology": "Curr. Biol.",
            "Trends in Neurosciences": "Trends Neurosci.",
            "Annual Review of Neuroscience": "Annu. Rev. Neurosci.",
            "Nature Reviews Neuroscience": "Nat. Rev. Neurosci.",
            "Frontiers in Neuroscience": "Front. Neurosci.",
            "Journal of Neurosurgery": "J. Neurosurg.",
            "Neurosurgery": "Neurosurgery",
            "Acta Neurochirurgica": "Acta Neurochir.",
            "World Neurosurgery": "World Neurosurg.",
            "Journal of Neuro-Oncology": "J. Neurooncol.",
            "Neuro-Oncology": "Neuro Oncol.",
            "Journal of Clinical Neuroscience": "J. Clin. Neurosci.",
            "Clinical Neurology and Neurosurgery": "Clin. Neurol. Neurosurg.",
            "Neurosurgical Focus": "Neurosurg Focus",
            "Journal of Neurology, Neurosurgery & Psychiatry": "J. Neurol. Neurosurg. Psychiatry"
        }
    
    def format_bibliography(self, references: List[Reference]) -> List[BibliographyEntry]:
        """
        Format complete bibliography according to APA style.
        
        Args:
            references: List of references to format
            
        Returns:
            List of formatted bibliography entries, sorted alphabetically
        """
        try:
            entries = []
            for ref in references:
                entry = self.format_reference(ref)
                entries.append(entry)
            
            # Sort entries by sort key
            entries.sort(key=lambda x: x.sort_key)
            
            logger.info(f"Formatted {len(entries)} bibliography entries")
            return entries
            
        except Exception as e:
            logger.error(f"Failed to format bibliography: {e}")
            raise APAFormattingError(f"Bibliography formatting failed: {e}")