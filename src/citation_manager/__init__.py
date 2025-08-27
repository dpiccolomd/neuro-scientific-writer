"""Citation management system for neuroscience literature."""

from .apa_formatter import APAFormatter
from .citation_extractor import CitationExtractor
from .zotero_integration import ZoteroConfig, ZoteroClient, ZoteroTrainingManager
from .models import (
    Citation, 
    Reference, 
    CitationContext, 
    ValidationResult,
    Author,
    Journal,
    CitationType,
    CitationStyle,
    ValidationStatus
)

__all__ = [
    'APAFormatter',
    'CitationExtractor', 
    'ZoteroConfig',
    'ZoteroClient',
    'ZoteroTrainingManager',
    'Citation',
    'Reference',
    'CitationContext',
    'ValidationResult',
    'Author',
    'Journal',
    'CitationType',
    'CitationStyle',
    'ValidationStatus'
]