"""
Reference Analyzer for Zotero Papers

Provides advanced analysis of reference papers from Zotero collections,
extracting key concepts, methodologies, and findings for citation integration.
"""

import logging
import re
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .models import Reference
from .zotero_integration import ZoteroClient, ZoteroItem
from pdf_processor import PDFExtractor
from pdf_processor.models import ProcessedDocument

logger = logging.getLogger(__name__)


@dataclass
class ConceptExtraction:
    """Extracted concepts from a reference paper."""
    neuroscience_concepts: List[str]
    methodological_concepts: List[str]
    clinical_concepts: List[str]
    statistical_concepts: List[str]
    confidence_score: float


@dataclass
class FindingExtraction:
    """Extracted findings from a reference paper."""
    primary_findings: List[str]
    secondary_findings: List[str]
    statistical_results: List[str]
    clinical_implications: List[str]
    confidence_score: float


@dataclass
class MethodologyExtraction:
    """Extracted methodological information."""
    techniques_used: List[str]
    data_analysis_methods: List[str]
    study_design: Optional[str]
    sample_characteristics: Optional[str]
    limitations_reported: List[str]
    confidence_score: float


@dataclass
class ReferenceAnalysisResult:
    """Complete analysis result for a reference paper."""
    reference: Reference
    concept_extraction: ConceptExtraction
    finding_extraction: FindingExtraction
    methodology_extraction: MethodologyExtraction
    research_gap_identified: Optional[str]
    theoretical_contribution: Optional[str]
    clinical_relevance_score: float
    methodological_novelty_score: float
    overall_quality_score: float


class ZoteroReferenceAnalyzer:
    """Analyzes reference papers from Zotero collections."""
    
    def __init__(self, zotero_client: ZoteroClient):
        """Initialize reference analyzer."""
        self.zotero_client = zotero_client
        self.pdf_extractor = PDFExtractor()
        self.neuroscience_vocabulary = self._load_neuroscience_vocabulary()
        self.methodology_vocabulary = self._load_methodology_vocabulary()
        self.clinical_vocabulary = self._load_clinical_vocabulary()
        
    def analyze_collection_references(
        self,
        collection_name: str,
        max_references: int = 20,
        download_pdfs: bool = True
    ) -> List[ReferenceAnalysisResult]:
        """
        Analyze all references in a Zotero collection.
        
        Args:
            collection_name: Name of Zotero collection
            max_references: Maximum number of references to analyze
            download_pdfs: Whether to download and process PDFs
            
        Returns:
            List of reference analysis results
        """
        logger.info(f"Analyzing references in collection: {collection_name}")
        
        # Get collection items
        items = self.zotero_client.get_collection_items(collection_name, limit=max_references * 2)
        
        analysis_results = []
        processed_count = 0
        
        for item in items:
            if processed_count >= max_references:
                break
                
            try:
                # Convert to Reference object
                reference = self._convert_zotero_item_to_reference(item)
                
                # Get full text if possible
                full_text = self._extract_full_text(item, download_pdfs)
                
                # Analyze reference
                analysis = self._analyze_single_reference(reference, full_text)
                analysis_results.append(analysis)
                
                processed_count += 1
                logger.debug(f"Analyzed {processed_count}/{max_references}: {reference.title[:50]}...")
                
            except Exception as e:
                logger.warning(f"Failed to analyze reference {item.title}: {e}")
                continue
        
        # Sort by overall quality score
        analysis_results.sort(key=lambda x: x.overall_quality_score, reverse=True)
        
        logger.info(f"Completed analysis of {len(analysis_results)} references")
        return analysis_results
    
    def analyze_specific_references(
        self,
        references: List[Reference],
        processed_documents: Optional[List[ProcessedDocument]] = None
    ) -> List[ReferenceAnalysisResult]:
        """
        Analyze specific reference papers.
        
        Args:
            references: List of references to analyze
            processed_documents: Pre-processed PDF documents (optional)
            
        Returns:
            List of reference analysis results
        """
        logger.info(f"Analyzing {len(references)} specific references")
        
        analysis_results = []
        
        for i, reference in enumerate(references):
            try:
                # Find corresponding processed document
                processed_doc = self._find_processed_document(reference, processed_documents)
                full_text = processed_doc.full_text if processed_doc else None
                
                # If no processed document, try to extract text from metadata
                if not full_text:
                    full_text = self._extract_text_from_reference_metadata(reference)
                
                # Analyze reference
                analysis = self._analyze_single_reference(reference, full_text)
                analysis_results.append(analysis)
                
                logger.debug(f"Analyzed {i+1}/{len(references)}: "
                           f"quality={analysis.overall_quality_score:.3f}")
                
            except Exception as e:
                logger.warning(f"Failed to analyze reference {reference.title}: {e}")
                continue
        
        return analysis_results
    
    def _analyze_single_reference(
        self, 
        reference: Reference, 
        full_text: Optional[str]
    ) -> ReferenceAnalysisResult:
        """Analyze a single reference paper."""
        
        # Combine available text sources
        analysis_text = self._combine_text_sources(reference, full_text)
        
        # Extract concepts
        concept_extraction = self._extract_concepts(analysis_text)
        
        # Extract findings
        finding_extraction = self._extract_findings(analysis_text)
        
        # Extract methodology
        methodology_extraction = self._extract_methodology(analysis_text, reference)
        
        # Identify research gaps
        research_gap = self._identify_research_gap(analysis_text)
        
        # Identify theoretical contributions
        theoretical_contribution = self._identify_theoretical_contribution(analysis_text)
        
        # Calculate relevance scores
        clinical_relevance = self._calculate_clinical_relevance(analysis_text, reference)
        methodological_novelty = self._calculate_methodological_novelty(methodology_extraction)
        overall_quality = self._calculate_overall_quality(
            concept_extraction, finding_extraction, methodology_extraction,
            clinical_relevance, methodological_novelty
        )
        
        return ReferenceAnalysisResult(
            reference=reference,
            concept_extraction=concept_extraction,
            finding_extraction=finding_extraction,
            methodology_extraction=methodology_extraction,
            research_gap_identified=research_gap,
            theoretical_contribution=theoretical_contribution,
            clinical_relevance_score=clinical_relevance,
            methodological_novelty_score=methodological_novelty,
            overall_quality_score=overall_quality
        )
    
    def _extract_concepts(self, text: str) -> ConceptExtraction:
        """Extract neuroscience concepts from text."""
        if not text:
            return ConceptExtraction([], [], [], [], 0.1)
        
        text_lower = text.lower()
        
        # Extract neuroscience concepts
        neuro_concepts = [
            concept for concept in self.neuroscience_vocabulary
            if concept.lower() in text_lower
        ]
        
        # Extract methodological concepts
        method_concepts = [
            concept for concept in self.methodology_vocabulary
            if concept.lower() in text_lower
        ]
        
        # Extract clinical concepts
        clinical_concepts = [
            concept for concept in self.clinical_vocabulary
            if concept.lower() in text_lower
        ]
        
        # Extract statistical concepts
        statistical_concepts = self._extract_statistical_concepts(text)
        
        # Calculate confidence based on text length and concept diversity
        confidence = self._calculate_concept_extraction_confidence(
            text, neuro_concepts, method_concepts, clinical_concepts
        )
        
        return ConceptExtraction(
            neuroscience_concepts=list(set(neuro_concepts))[:15],
            methodological_concepts=list(set(method_concepts))[:10],
            clinical_concepts=list(set(clinical_concepts))[:10],
            statistical_concepts=statistical_concepts,
            confidence_score=confidence
        )
    
    def _extract_findings(self, text: str) -> FindingExtraction:
        """Extract research findings from text."""
        if not text:
            return FindingExtraction([], [], [], [], 0.1)
        
        # Patterns for identifying findings
        finding_patterns = [
            r"we found that ([^.]{20,150})",
            r"results showed ([^.]{20,150})",
            r"our findings indicate ([^.]{20,150})",
            r"the data revealed ([^.]{20,150})",
            r"analysis demonstrated ([^.]{20,150})",
            r"we observed ([^.]{20,150})"
        ]
        
        primary_findings = []
        for pattern in finding_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            primary_findings.extend([match.strip() for match in matches[:2]])
        
        # Extract secondary findings
        secondary_patterns = [
            r"additionally, ([^.]{20,150})",
            r"furthermore, ([^.]{20,150})",
            r"we also found ([^.]{20,150})",
            r"secondary analysis revealed ([^.]{20,150})"
        ]
        
        secondary_findings = []
        for pattern in secondary_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            secondary_findings.extend([match.strip() for match in matches[:2]])
        
        # Extract statistical results
        statistical_results = self._extract_statistical_results(text)
        
        # Extract clinical implications
        clinical_implications = self._extract_clinical_implications(text)
        
        confidence = min(
            (len(primary_findings) + len(secondary_findings)) / 5.0,
            1.0
        )
        
        return FindingExtraction(
            primary_findings=primary_findings[:5],
            secondary_findings=secondary_findings[:3],
            statistical_results=statistical_results[:5],
            clinical_implications=clinical_implications[:3],
            confidence_score=confidence
        )
    
    def _extract_methodology(self, text: str, reference: Reference) -> MethodologyExtraction:
        """Extract methodological information from text."""
        if not text:
            return MethodologyExtraction([], [], None, None, [], 0.1)
        
        # Extract techniques used
        techniques = [
            technique for technique in self.methodology_vocabulary
            if technique.lower() in text.lower()
        ]
        
        # Extract data analysis methods
        analysis_methods = self._extract_analysis_methods(text)
        
        # Extract study design
        study_design = self._extract_study_design(text)
        
        # Extract sample characteristics
        sample_characteristics = self._extract_sample_characteristics(text)
        
        # Extract limitations
        limitations = self._extract_limitations(text)
        
        confidence = min(
            (len(techniques) + len(analysis_methods) + len(limitations)) / 8.0,
            1.0
        )
        
        return MethodologyExtraction(
            techniques_used=list(set(techniques))[:8],
            data_analysis_methods=analysis_methods[:5],
            study_design=study_design,
            sample_characteristics=sample_characteristics,
            limitations_reported=limitations[:5],
            confidence_score=confidence
        )
    
    def _extract_statistical_results(self, text: str) -> List[str]:
        """Extract statistical results from text."""
        # Patterns for statistical results
        stat_patterns = [
            r"p\s*[<>=]\s*0?\.\d+",
            r"r\s*=\s*0?\.\d+",
            r"t\s*=\s*-?\d+\.\d+",
            r"F\s*=\s*\d+\.\d+",
            r"χ²\s*=\s*\d+\.\d+",
            r"d\s*=\s*0?\.\d+",
            r"\d+%\s*(increase|decrease|change)",
            r"significant.*p\s*<\s*0?\.\d+"
        ]
        
        results = []
        for pattern in stat_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            results.extend(matches[:3])
        
        return list(set(results))[:8]
    
    def _extract_clinical_implications(self, text: str) -> List[str]:
        """Extract clinical implications from text."""
        impl_patterns = [
            r"clinical implications? ([^.]{20,150})",
            r"therapeutic potential ([^.]{20,150})",
            r"treatment implications? ([^.]{20,150})",
            r"patient outcomes? ([^.]{20,150})",
            r"clinical significance ([^.]{20,150})"
        ]
        
        implications = []
        for pattern in impl_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            implications.extend([match.strip() for match in matches[:2]])
        
        return implications[:5]
    
    def _identify_research_gap(self, text: str) -> Optional[str]:
        """Identify research gaps mentioned in the text."""
        gap_patterns = [
            r"gap in understanding ([^.]{20,150})",
            r"remains unclear ([^.]{20,150})",
            r"poorly understood ([^.]{20,150})",
            r"further research ([^.]{20,150})",
            r"future studies ([^.]{20,150})",
            r"limitations? of current ([^.]{20,150})"
        ]
        
        for pattern in gap_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _calculate_clinical_relevance(self, text: str, reference: Reference) -> float:
        """Calculate clinical relevance score."""
        score = 0.0
        
        if not text:
            return 0.3
        
        text_lower = text.lower()
        
        # Clinical keywords
        clinical_keywords = [
            'patient', 'clinical', 'treatment', 'therapy', 'diagnosis',
            'prognosis', 'intervention', 'therapeutic', 'medical', 'hospital'
        ]
        
        clinical_mentions = sum(1 for keyword in clinical_keywords if keyword in text_lower)
        score += min(clinical_mentions / 10.0, 0.4)
        
        # Journal factor
        if reference.journal:
            clinical_journals = ['clinical', 'medicine', 'medical', 'therapy', 'treatment']
            if any(journal_word in reference.journal.lower() for journal_word in clinical_journals):
                score += 0.3
        
        # Study type indicators
        clinical_study_types = ['trial', 'intervention', 'treatment', 'therapy']
        study_type_mentions = sum(1 for study_type in clinical_study_types if study_type in text_lower)
        score += min(study_type_mentions / 4.0, 0.3)
        
        return min(score, 1.0)
    
    def _calculate_methodological_novelty(self, methodology: MethodologyExtraction) -> float:
        """Calculate methodological novelty score."""
        if not methodology.techniques_used:
            return 0.3
        
        # Check for novel/advanced techniques
        novel_techniques = [
            'optogenetics', 'two-photon', 'single-cell', 'crispr', 'machine learning',
            'deep learning', 'artificial intelligence', 'real-time', 'multimodal'
        ]
        
        novel_count = sum(
            1 for technique in methodology.techniques_used
            if any(novel_term in technique.lower() for novel_term in novel_techniques)
        )
        
        novelty_score = min(novel_count / len(methodology.techniques_used), 0.8)
        
        # Boost for multiple techniques
        if len(methodology.techniques_used) > 3:
            novelty_score += 0.2
        
        return min(novelty_score, 1.0)
    
    def _calculate_overall_quality(
        self,
        concepts: ConceptExtraction,
        findings: FindingExtraction,
        methodology: MethodologyExtraction,
        clinical_relevance: float,
        methodological_novelty: float
    ) -> float:
        """Calculate overall quality score for reference."""
        
        # Weight different factors
        quality_score = (
            concepts.confidence_score * 0.2 +
            findings.confidence_score * 0.25 +
            methodology.confidence_score * 0.25 +
            clinical_relevance * 0.15 +
            methodological_novelty * 0.15
        )
        
        return min(quality_score, 1.0)
    
    # Helper methods
    def _convert_zotero_item_to_reference(self, item: ZoteroItem) -> Reference:
        """Convert Zotero item to Reference object."""
        return Reference(
            title=item.title,
            authors=item.authors,
            publication_year=item.publication_year,
            journal=item.journal,
            doi=item.doi,
            abstract=item.abstract,
            url=item.url
        )
    
    def _extract_full_text(self, item: ZoteroItem, download_pdfs: bool) -> Optional[str]:
        """Extract full text from Zotero item."""
        if not download_pdfs or not item.pdf_attachments:
            return None
        
        # This would download and process PDFs
        # For now, return None (would implement PDF processing)
        return None
    
    def _combine_text_sources(self, reference: Reference, full_text: Optional[str]) -> str:
        """Combine all available text sources."""
        text_parts = []
        
        if full_text:
            text_parts.append(full_text[:5000])  # Limit to prevent memory issues
        
        if reference.title:
            text_parts.append(reference.title)
        
        if reference.abstract:
            text_parts.append(reference.abstract)
        
        return " ".join(text_parts)
    
    def _find_processed_document(
        self, 
        reference: Reference, 
        processed_docs: Optional[List[ProcessedDocument]]
    ) -> Optional[ProcessedDocument]:
        """Find processed document matching reference."""
        if not processed_docs:
            return None
        
        for doc in processed_docs:
            # Simple matching by title similarity
            if (reference.title and doc.title and 
                reference.title.lower() in doc.title.lower()):
                return doc
        
        return None
    
    def _extract_text_from_reference_metadata(self, reference: Reference) -> str:
        """Extract text from reference metadata only."""
        parts = []
        if reference.title:
            parts.append(reference.title)
        if reference.abstract:
            parts.append(reference.abstract)
        return " ".join(parts)
    
    def _load_neuroscience_vocabulary(self) -> List[str]:
        """Load neuroscience vocabulary for concept extraction."""
        return [
            # Core neuroscience terms
            "neuroplasticity", "synaptic plasticity", "long-term potentiation", "neurotransmitter",
            "dopamine", "serotonin", "GABA", "glutamate", "acetylcholine", "norepinephrine",
            "cortical", "hippocampus", "amygdala", "prefrontal cortex", "temporal lobe",
            "parietal lobe", "occipital lobe", "cerebellum", "brainstem", "thalamus",
            "neural network", "connectivity", "white matter", "gray matter", "myelin",
            "axon", "dendrite", "soma", "synapse", "action potential", "membrane potential",
            "ion channels", "receptors", "neurodevelopment", "neurodegeneration",
            "cognitive function", "executive function", "working memory", "episodic memory",
            "semantic memory", "attention", "consciousness", "perception", "learning",
            "motor control", "sensory processing", "language", "emotion", "reward",
            "circadian rhythm", "sleep", "arousal", "stress response"
        ]
    
    def _load_methodology_vocabulary(self) -> List[str]:
        """Load methodology vocabulary."""
        return [
            # Neuroimaging
            "fMRI", "functional magnetic resonance imaging", "structural MRI", "DTI",
            "diffusion tensor imaging", "PET", "positron emission tomography",
            "EEG", "electroencephalography", "MEG", "magnetoencephalography",
            "NIRS", "near-infrared spectroscopy",
            
            # Electrophysiology
            "patch clamp", "whole-cell recording", "extracellular recording",
            "single-unit recording", "local field potential", "ERP", "event-related potential",
            
            # Optical techniques
            "two-photon microscopy", "confocal microscopy", "calcium imaging",
            "optogenetics", "fluorescence microscopy", "live imaging",
            
            # Molecular techniques
            "immunohistochemistry", "immunofluorescence", "Western blot",
            "RT-PCR", "qPCR", "RNA sequencing", "single-cell RNA-seq",
            "proteomics", "metabolomics", "ChIP-seq", "ATAC-seq",
            
            # Genetic approaches
            "transgenic", "knockout", "knock-in", "CRISPR", "viral vectors",
            "optogenetics", "chemogenetics", "DREADDs",
            
            # Behavioral methods
            "behavioral testing", "cognitive assessment", "neuropsychological testing",
            "operant conditioning", "fear conditioning", "Morris water maze",
            "open field test", "elevated plus maze",
            
            # Analysis methods
            "machine learning", "deep learning", "artificial neural networks",
            "computational modeling", "statistical analysis", "meta-analysis"
        ]
    
    def _load_clinical_vocabulary(self) -> List[str]:
        """Load clinical vocabulary."""
        return [
            "clinical trial", "randomized controlled trial", "case-control study",
            "cohort study", "patient", "treatment", "therapy", "intervention",
            "diagnosis", "prognosis", "biomarker", "therapeutic target",
            "drug discovery", "pharmacology", "side effects", "adverse events",
            "quality of life", "functional outcomes", "disability", "rehabilitation",
            "neurological disorders", "psychiatric disorders", "neurodegenerative disease",
            "stroke", "epilepsy", "Alzheimer's disease", "Parkinson's disease",
            "depression", "anxiety", "schizophrenia", "autism", "ADHD",
            "traumatic brain injury", "spinal cord injury", "multiple sclerosis"
        ]
    
    def _extract_statistical_concepts(self, text: str) -> List[str]:
        """Extract statistical concepts and methods."""
        stat_terms = [
            "ANOVA", "t-test", "regression", "correlation", "chi-square",
            "Mann-Whitney", "Wilcoxon", "Kruskal-Wallis", "effect size",
            "confidence interval", "p-value", "significance", "power analysis",
            "multiple comparisons", "Bonferroni", "FDR", "false discovery rate",
            "bootstrapping", "permutation test", "Bayesian", "cross-validation"
        ]
        
        found_terms = []
        text_lower = text.lower()
        
        for term in stat_terms:
            if term.lower() in text_lower:
                found_terms.append(term)
        
        return found_terms[:10]
    
    def _calculate_concept_extraction_confidence(
        self, 
        text: str, 
        neuro_concepts: List[str],
        method_concepts: List[str], 
        clinical_concepts: List[str]
    ) -> float:
        """Calculate confidence in concept extraction."""
        if not text:
            return 0.1
        
        # Text length factor
        text_length_factor = min(len(text) / 1000.0, 1.0)
        
        # Concept diversity factor
        total_concepts = len(neuro_concepts) + len(method_concepts) + len(clinical_concepts)
        concept_factor = min(total_concepts / 15.0, 1.0)
        
        return (text_length_factor * 0.6 + concept_factor * 0.4)
    
    def _extract_analysis_methods(self, text: str) -> List[str]:
        """Extract data analysis methods mentioned in text."""
        analysis_terms = [
            "statistical analysis", "data analysis", "multivariate analysis",
            "univariate analysis", "time-series analysis", "network analysis",
            "connectivity analysis", "machine learning", "classification",
            "clustering", "dimensionality reduction", "PCA", "ICA"
        ]
        
        found_methods = []
        text_lower = text.lower()
        
        for method in analysis_terms:
            if method.lower() in text_lower:
                found_methods.append(method)
        
        return found_methods
    
    def _extract_study_design(self, text: str) -> Optional[str]:
        """Extract study design information."""
        design_patterns = [
            r"(randomized controlled trial|RCT)",
            r"(case-control study)",
            r"(cohort study)",
            r"(cross-sectional study)",
            r"(longitudinal study)",
            r"(experimental study)",
            r"(observational study)"
        ]
        
        for pattern in design_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_sample_characteristics(self, text: str) -> Optional[str]:
        """Extract sample characteristics."""
        sample_patterns = [
            r"(\d+\s+participants?)",
            r"(\d+\s+subjects?)",
            r"(n\s*=\s*\d+)",
            r"(age range \d+-\d+)",
            r"(mean age \d+\.\d+)"
        ]
        
        characteristics = []
        for pattern in sample_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            characteristics.extend(matches)
        
        return "; ".join(characteristics[:3]) if characteristics else None
    
    def _extract_limitations(self, text: str) -> List[str]:
        """Extract study limitations mentioned in text."""
        limitation_patterns = [
            r"limitations? include ([^.]{20,150})",
            r"limited by ([^.]{20,150})",
            r"constraint[s]? ([^.]{20,150})",
            r"caveat[s]? ([^.]{20,150})",
            r"should be interpreted with caution ([^.]{20,150})"
        ]
        
        limitations = []
        for pattern in limitation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            limitations.extend([match.strip() for match in matches[:2]])
        
        return limitations[:5]