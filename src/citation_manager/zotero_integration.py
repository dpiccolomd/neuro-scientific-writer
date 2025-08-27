"""
Zotero Integration Module

Provides seamless integration with Zotero reference manager for automated
PDF collection and metadata extraction for empirical pattern training.

This module connects to user's Zotero library, filters neuroscience papers,
downloads PDFs, and preserves metadata for empirical analysis.
"""

import logging
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time

from .models import Reference, Author, Journal, CitationType
from .exceptions import ZoteroIntegrationError

logger = logging.getLogger(__name__)


@dataclass
class ZoteroConfig:
    """Zotero configuration settings."""
    api_key: str
    user_id: Optional[str] = None
    group_id: Optional[str] = None
    library_type: str = "user"  # "user" or "group"
    base_url: str = "https://api.zotero.org"
    
    def __post_init__(self):
        if not self.api_key:
            raise ValueError("Zotero API key is required")
        if self.library_type == "user" and not self.user_id:
            raise ValueError("User ID required for user library")
        if self.library_type == "group" and not self.group_id:
            raise ValueError("Group ID required for group library")


@dataclass
class ZoteroItem:
    """Zotero library item with metadata."""
    item_key: str
    item_type: str
    title: str
    authors: List[str]
    publication_year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    collections: List[str] = field(default_factory=list)
    pdf_attachments: List[str] = field(default_factory=list)
    url: Optional[str] = None
    
    @property
    def is_neuroscience_related(self) -> bool:
        """Check if item is neuroscience-related based on content."""
        neuroscience_keywords = [
            'neuroscience', 'neurosurgery', 'brain', 'neural', 'neuron',
            'cortex', 'cognitive', 'fmri', 'neuroimaging', 'plasticity',
            'neurology', 'neurological', 'neurotransmitter', 'synapse'
        ]
        
        # Check title
        text_to_check = (self.title or '').lower()
        
        # Check journal name
        if self.journal:
            text_to_check += ' ' + self.journal.lower()
        
        # Check abstract
        if self.abstract:
            text_to_check += ' ' + self.abstract.lower()
        
        # Check tags
        tag_text = ' '.join(self.tags).lower()
        text_to_check += ' ' + tag_text
        
        return any(keyword in text_to_check for keyword in neuroscience_keywords)


class ZoteroClient:
    """Client for interacting with Zotero Web API."""
    
    def __init__(self, config: ZoteroConfig):
        """Initialize Zotero client with configuration."""
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'Zotero-API-Key': config.api_key,
            'User-Agent': 'Neuro-Scientific-Writer/1.0'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make rate-limited request to Zotero API."""
        # Rate limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        library_path = f"users/{self.config.user_id}" if self.config.library_type == "user" \
                      else f"groups/{self.config.group_id}"
        
        url = f"{self.config.base_url}/{library_path}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            self.last_request_time = time.time()
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"Zotero API request failed: {e}")
            raise ZoteroIntegrationError(f"API request failed: {e}")
    
    def get_collections(self) -> List[Dict]:
        """Get all collections from the library."""
        logger.info("Fetching Zotero collections")
        return self._make_request("collections")
    
    def get_collection_items(self, collection_key: str, limit: int = 100) -> List[ZoteroItem]:
        """Get items from a specific collection."""
        logger.info(f"Fetching items from collection {collection_key}")
        
        params = {
            'format': 'json',
            'limit': limit,
            'itemType': 'journalArticle'  # Focus on journal articles
        }
        
        items_data = self._make_request(f"collections/{collection_key}/items", params)
        items = []
        
        for item_data in items_data:
            try:
                item = self._parse_zotero_item(item_data)
                if item:
                    items.append(item)
            except Exception as e:
                logger.warning(f"Failed to parse item {item_data.get('key', 'unknown')}: {e}")
        
        logger.info(f"Retrieved {len(items)} items from collection")
        return items
    
    def get_all_items(self, limit: int = 200) -> List[ZoteroItem]:
        """Get all items from the library."""
        logger.info("Fetching all items from library")
        
        params = {
            'format': 'json',
            'limit': limit,
            'itemType': 'journalArticle'
        }
        
        items_data = self._make_request("items", params)
        items = []
        
        for item_data in items_data:
            try:
                item = self._parse_zotero_item(item_data)
                if item and item.is_neuroscience_related:
                    items.append(item)
            except Exception as e:
                logger.warning(f"Failed to parse item {item_data.get('key', 'unknown')}: {e}")
        
        logger.info(f"Retrieved {len(items)} neuroscience-related items")
        return items
    
    def download_pdf_attachment(self, attachment_key: str, output_path: Path) -> bool:
        """Download PDF attachment from Zotero."""
        try:
            # Get attachment file
            endpoint = f"items/{attachment_key}/file"
            url = f"{self.config.base_url}/users/{self.config.user_id}/{endpoint}" \
                  if self.config.library_type == "user" \
                  else f"{self.config.base_url}/groups/{self.config.group_id}/{endpoint}"
            
            response = self.session.get(url)
            response.raise_for_status()
            
            # Save PDF
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded PDF: {output_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download attachment {attachment_key}: {e}")
            return False
    
    def _parse_zotero_item(self, item_data: Dict) -> Optional[ZoteroItem]:
        """Parse Zotero item data into ZoteroItem object."""
        try:
            data = item_data.get('data', {})
            
            # Extract basic information
            item = ZoteroItem(
                item_key=item_data.get('key', ''),
                item_type=data.get('itemType', ''),
                title=data.get('title', ''),
                authors=self._extract_authors(data.get('creators', [])),
                publication_year=self._extract_year(data.get('date', '')),
                journal=data.get('publicationTitle', ''),
                doi=data.get('DOI', ''),
                abstract=data.get('abstractNote', ''),
                tags=[tag.get('tag', '') for tag in data.get('tags', [])],
                url=data.get('url', '')
            )
            
            # Get PDF attachments
            if 'children' in item_data:
                for child in item_data['children']:
                    if (child.get('data', {}).get('itemType') == 'attachment' and
                        child.get('data', {}).get('contentType') == 'application/pdf'):
                        item.pdf_attachments.append(child.get('key', ''))
            
            return item
            
        except Exception as e:
            logger.warning(f"Failed to parse Zotero item: {e}")
            return None
    
    def _extract_authors(self, creators: List[Dict]) -> List[str]:
        """Extract author names from Zotero creators."""
        authors = []
        for creator in creators:
            if creator.get('creatorType') == 'author':
                first_name = creator.get('firstName', '')
                last_name = creator.get('lastName', '')
                if last_name:
                    name = f"{last_name}, {first_name}" if first_name else last_name
                    authors.append(name)
        return authors
    
    def _extract_year(self, date_string: str) -> Optional[int]:
        """Extract year from Zotero date string."""
        if not date_string:
            return None
        
        # Try to extract 4-digit year
        import re
        year_match = re.search(r'\b(19|20)\d{2}\b', date_string)
        if year_match:
            return int(year_match.group())
        
        return None


class ZoteroTrainingManager:
    """Manages training data collection from Zotero libraries."""
    
    def __init__(self, config: ZoteroConfig, output_dir: str = "data/training_papers"):
        """Initialize training manager."""
        self.client = ZoteroClient(config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file
        self.metadata_file = self.output_dir / "zotero_metadata.json"
        self.metadata = self._load_metadata()
    
    def collect_training_papers(
        self, 
        collection_name: Optional[str] = None,
        min_papers: int = 50,
        max_papers: int = 200
    ) -> Dict:
        """
        Collect training papers from Zotero library.
        
        Args:
            collection_name: Specific collection to use, or None for entire library
            min_papers: Minimum papers required
            max_papers: Maximum papers to collect
            
        Returns:
            Collection results with statistics
        """
        logger.info(f"Starting Zotero training paper collection")
        
        # Get items
        if collection_name:
            items = self._get_items_from_collection(collection_name)
        else:
            items = self.client.get_all_items(limit=max_papers)
        
        # Filter neuroscience papers with PDFs
        valid_items = [item for item in items if item.is_neuroscience_related and item.pdf_attachments]
        
        if len(valid_items) < min_papers:
            raise ZoteroIntegrationError(
                f"Insufficient papers with PDFs: {len(valid_items)} < {min_papers}. "
                f"Need more neuroscience papers with PDF attachments in your Zotero library."
            )
        
        # Limit to max_papers
        if len(valid_items) > max_papers:
            valid_items = valid_items[:max_papers]
            logger.warning(f"Limited to {max_papers} papers for performance")
        
        # Download PDFs and collect metadata
        results = {
            'total_items_found': len(items),
            'neuroscience_papers': len(valid_items),
            'successful_downloads': 0,
            'failed_downloads': 0,
            'papers_metadata': [],
            'collection_date': datetime.now().isoformat()
        }
        
        for i, item in enumerate(valid_items, 1):
            logger.info(f"Processing {i}/{len(valid_items)}: {item.title[:50]}...")
            
            success = False
            for attachment_key in item.pdf_attachments:
                pdf_filename = f"{item.item_key}_{attachment_key}.pdf"
                pdf_path = self.output_dir / pdf_filename
                
                if self.client.download_pdf_attachment(attachment_key, pdf_path):
                    success = True
                    results['successful_downloads'] += 1
                    
                    # Save metadata
                    paper_metadata = {
                        'zotero_key': item.item_key,
                        'title': item.title,
                        'authors': item.authors,
                        'journal': item.journal,
                        'year': item.publication_year,
                        'doi': item.doi,
                        'pdf_file': pdf_filename,
                        'tags': item.tags,
                        'abstract': item.abstract
                    }
                    
                    results['papers_metadata'].append(paper_metadata)
                    self.metadata[item.item_key] = paper_metadata
                    break
            
            if not success:
                results['failed_downloads'] += 1
        
        # Save metadata
        self._save_metadata()
        
        logger.info(f"Collection complete: {results['successful_downloads']} papers downloaded")
        return results
    
    def _get_items_from_collection(self, collection_name: str) -> List[ZoteroItem]:
        """Get items from a specific collection by name."""
        collections = self.client.get_collections()
        
        collection_key = None
        for collection in collections:
            if collection.get('data', {}).get('name') == collection_name:
                collection_key = collection.get('key')
                break
        
        if not collection_key:
            available = [c.get('data', {}).get('name', 'Unknown') for c in collections]
            raise ZoteroIntegrationError(
                f"Collection '{collection_name}' not found. "
                f"Available collections: {', '.join(available)}"
            )
        
        return self.client.get_collection_items(collection_key)
    
    def _load_metadata(self) -> Dict:
        """Load existing metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save metadata to file."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def get_collection_summary(self) -> Dict:
        """Get summary of collected papers."""
        pdf_files = list(self.output_dir.glob("*.pdf"))
        
        # Analyze journals
        journals = {}
        for paper in self.metadata.values():
            journal = paper.get('journal', 'Unknown')
            journals[journal] = journals.get(journal, 0) + 1
        
        # Analyze years
        years = {}
        for paper in self.metadata.values():
            year = paper.get('year', 'Unknown')
            years[str(year)] = years.get(str(year), 0) + 1
        
        return {
            'total_pdfs': len(pdf_files),
            'total_metadata_entries': len(self.metadata),
            'journals_represented': len(journals),
            'top_journals': dict(sorted(journals.items(), key=lambda x: x[1], reverse=True)[:10]),
            'year_distribution': dict(sorted(years.items(), key=lambda x: x[1], reverse=True)[:10])
        }