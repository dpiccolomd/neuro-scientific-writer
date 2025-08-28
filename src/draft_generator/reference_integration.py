"""
Reference Integration System

Analyzes reference papers and creates contextual integration plans
for citation-aware introduction generation.
"""

import logging
import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .models import ReferenceContext, ReferenceRole, StudySpecification
from citation_manager.models import Reference
from citation_manager.apa_formatter import APAFormatter
from pdf_processor.models import ProcessedDocument

logger = logging.getLogger(__name__)


@dataclass
class ReferenceAnalysis:
    """Analysis of a single reference paper for integration planning."""
    reference: Reference
    key_concepts: List[str]
    key_findings: List[str]
    methodological_approaches: List[str]
    limitations_identified: List[str]
    research_gap_identified: Optional[str]
    theoretical_framework: Optional[str]
    clinical_significance: Optional[str]
    relevance_score: float  # 0-1 relevance to target study
    recommended_role: ReferenceRole
    confidence_score: float  # Confidence in the analysis


class ReferenceIntegrator:
    """
    Analyzes reference papers and creates integration plans for
    citation-aware introduction generation.
    """
    
    def __init__(self):
        """Initialize reference integrator."""
        self.apa_formatter = APAFormatter()
        self.neuroscience_concepts = self._load_neuroscience_concepts()
        self.methodology_keywords = self._load_methodology_keywords()
        
    def analyze_references_for_study(
        self,
        references: List[Reference],
        study_spec: StudySpecification,
        processed_documents: Optional[List[ProcessedDocument]] = None
    ) -> List[ReferenceAnalysis]:
        """
        Analyze references for their relevance to the target study.
        
        Args:
            references: List of reference papers
            study_spec: Target study specification
            processed_documents: Processed PDF documents (if available)
            
        Returns:
            List of reference analyses with integration recommendations
        """
        logger.info(f"Analyzing {len(references)} references for study integration")
        
        analyses = []
        for i, reference in enumerate(references):
            try:
                # Find corresponding processed document if available
                processed_doc = self._find_processed_document(reference, processed_documents)
                
                analysis = self._analyze_single_reference(
                    reference, study_spec, processed_doc
                )
                analyses.append(analysis)
                
                logger.debug(f"Analyzed reference {i+1}/{len(references)}: "
                           f"relevance={analysis.relevance_score:.3f}, "
                           f"role={analysis.recommended_role.value}")
                
            except Exception as e:
                logger.warning(f"Failed to analyze reference {reference.title}: {e}")
                # Create minimal analysis
                analyses.append(self._create_minimal_analysis(reference))
        
        # Sort by relevance score
        analyses.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"Completed reference analysis: "
                   f"avg_relevance={sum(a.relevance_score for a in analyses)/len(analyses):.3f}")
        
        return analyses
    
    def create_reference_contexts(
        self,
        analyses: List[ReferenceAnalysis],
        study_spec: StudySpecification,
        max_references: int = 15
    ) -> List[ReferenceContext]:
        """
        Create reference contexts for citation integration.
        
        Args:
            analyses: Reference analyses
            study_spec: Target study specification
            max_references: Maximum number of references to include
            
        Returns:
            List of reference contexts ready for integration
        """
        # Select top references by relevance
        selected_analyses = analyses[:max_references]
        
        contexts = []
        for analysis in selected_analyses:
            context = self._create_reference_context(analysis, study_spec)
            contexts.append(context)
        
        # Ensure balanced distribution of roles
        contexts = self._balance_reference_roles(contexts)
        
        # Assign target paragraphs based on roles
        contexts = self._assign_paragraph_targets(contexts)
        
        logger.info(f"Created {len(contexts)} reference contexts")
        self._log_role_distribution(contexts)
        
        return contexts
    
    def _analyze_single_reference(
        self,
        reference: Reference,
        study_spec: StudySpecification,
        processed_doc: Optional[ProcessedDocument] = None
    ) -> ReferenceAnalysis:
        """Analyze a single reference paper."""
        
        # Extract text for analysis
        text_content = self._extract_reference_text(reference, processed_doc)
        
        # Analyze key concepts
        key_concepts = self._extract_key_concepts(text_content, study_spec)
        
        # Analyze key findings
        key_findings = self._extract_key_findings(text_content)
        
        # Analyze methodological approaches
        methodological_approaches = self._extract_methodological_approaches(text_content)
        
        # Identify limitations
        limitations = self._extract_limitations(text_content)
        
        # Identify research gaps
        research_gap = self._extract_research_gap(text_content)
        
        # Extract theoretical framework
        theoretical_framework = self._extract_theoretical_framework(text_content)
        
        # Extract clinical significance
        clinical_significance = self._extract_clinical_significance(text_content)
        
        # Calculate relevance score
        relevance_score = self._calculate_relevance_score(
            reference, study_spec, key_concepts, key_findings
        )
        
        # Recommend role
        recommended_role = self._recommend_reference_role(
            reference, study_spec, key_concepts, methodological_approaches, research_gap
        )
        
        # Calculate confidence in analysis
        confidence_score = self._calculate_analysis_confidence(
            text_content, key_concepts, key_findings
        )
        
        return ReferenceAnalysis(
            reference=reference,
            key_concepts=key_concepts,
            key_findings=key_findings,
            methodological_approaches=methodological_approaches,
            limitations_identified=limitations,
            research_gap_identified=research_gap,
            theoretical_framework=theoretical_framework,
            clinical_significance=clinical_significance,
            relevance_score=relevance_score,
            recommended_role=recommended_role,
            confidence_score=confidence_score
        )
    
    def _extract_reference_text(
        self, 
        reference: Reference, 
        processed_doc: Optional[ProcessedDocument] = None
    ) -> str:
        """Extract text content from reference for analysis."""
        text_parts = []
        
        # Use processed document if available
        if processed_doc and processed_doc.full_text:
            return processed_doc.full_text[:5000]  # Limit to first 5000 chars
        
        # Otherwise use available metadata
        if reference.title:
            text_parts.append(reference.title)
        
        if reference.abstract:
            text_parts.append(reference.abstract)
        
        if reference.keywords:
            text_parts.extend(reference.keywords)
        
        return " ".join(text_parts)
    
    def _extract_key_concepts(self, text: str, study_spec: StudySpecification) -> List[str]:
        """Extract key neuroscience concepts from reference text."""
        if not text:
            return []
        
        text_lower = text.lower()
        concepts_found = []
        
        # Check for neuroscience concepts
        for concept in self.neuroscience_concepts:
            if concept.lower() in text_lower:
                concepts_found.append(concept)
        
        # Look for concepts related to study domain
        domain_keywords = self._get_domain_specific_keywords(study_spec.research_domain)
        for keyword in domain_keywords:
            if keyword.lower() in text_lower:
                concepts_found.append(keyword)
        
        return list(set(concepts_found))[:10]  # Limit to top 10
    
    def _extract_key_findings(self, text: str) -> List[str]:
        """Extract key findings from reference text."""
        if not text:
            return []
        
        findings = []
        
        # Look for common finding indicators
        finding_patterns = [
            r"we found that (.{10,100})",
            r"results showed (.{10,100})",
            r"demonstrated that (.{10,100})",
            r"revealed that (.{10,100})",
            r"indicated that (.{10,100})"
        ]
        
        for pattern in finding_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            findings.extend(matches[:3])  # Limit matches per pattern
        
        return findings[:8]  # Limit total findings
    
    def _extract_methodological_approaches(self, text: str) -> List[str]:
        """Extract methodological approaches from reference text."""
        if not text:
            return []
        
        text_lower = text.lower()
        methods_found = []
        
        for method in self.methodology_keywords:
            if method.lower() in text_lower:
                methods_found.append(method)
        
        return list(set(methods_found))[:8]
    
    def _extract_limitations(self, text: str) -> List[str]:
        """Extract limitations mentioned in reference text."""
        if not text:
            return []
        
        limitations = []
        
        # Look for limitation indicators
        limitation_patterns = [
            r"limitation[s]? include (.{10,100})",
            r"limited by (.{10,100})",
            r"constraint[s]? (.{10,100})",
            r"however, (.{10,100})",
            r"nevertheless, (.{10,100})"
        ]
        
        for pattern in limitation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            limitations.extend(matches[:2])
        
        return limitations[:5]
    
    def _extract_research_gap(self, text: str) -> Optional[str]:
        """Extract research gap identified in reference text."""
        if not text:
            return None
        
        gap_patterns = [
            r"gap in (.{10,100})",
            r"remain[s]? unclear (.{10,100})",
            r"poorly understood (.{10,100})",
            r"further research (.{10,100})",
            r"future studies (.{10,100})"
        ]
        
        for pattern in gap_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_theoretical_framework(self, text: str) -> Optional[str]:
        """Extract theoretical framework mentioned in reference."""
        if not text:
            return None
        
        framework_keywords = [
            "theoretical framework", "theory", "model", "framework",
            "approach", "paradigm", "hypothesis"
        ]
        
        for keyword in framework_keywords:
            if keyword in text.lower():
                # Find sentence containing the keyword
                sentences = text.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        return sentence.strip()[:200]
        
        return None
    
    def _extract_clinical_significance(self, text: str) -> Optional[str]:
        """Extract clinical significance from reference text."""
        if not text:
            return None
        
        clinical_patterns = [
            r"clinical (.{10,100})",
            r"patient[s]? (.{10,100})",
            r"treatment (.{10,100})",
            r"therapeutic (.{10,100})"
        ]
        
        for pattern in clinical_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).strip()[:150]
        
        return None
    
    def _calculate_relevance_score(
        self,
        reference: Reference,
        study_spec: StudySpecification,
        key_concepts: List[str],
        key_findings: List[str]
    ) -> float:
        """Calculate relevance score of reference to target study."""
        score = 0.0
        
        # Domain match
        if study_spec.research_domain.replace('_', ' ') in (reference.title or '').lower():
            score += 0.3
        
        # Concept overlap
        study_concepts = self._extract_study_concepts(study_spec)
        concept_overlap = len(set(key_concepts) & set(study_concepts))
        score += min(concept_overlap * 0.1, 0.4)
        
        # Journal quality (if neuroscience journal)
        if reference.journal and any(
            neuro_word in reference.journal.lower() 
            for neuro_word in ['neuroscience', 'brain', 'neural', 'neurology']
        ):
            score += 0.2
        
        # Publication recency
        if reference.publication_year and reference.publication_year >= 2015:
            score += 0.1
        
        return min(score, 1.0)
    
    def _recommend_reference_role(
        self,
        reference: Reference,
        study_spec: StudySpecification,
        key_concepts: List[str],
        methodological_approaches: List[str],
        research_gap: Optional[str]
    ) -> ReferenceRole:
        """Recommend role for reference based on content analysis."""
        
        # Check if it's a foundational/review paper
        if reference.title and any(
            word in reference.title.lower() 
            for word in ['review', 'overview', 'survey', 'foundations']
        ):
            return ReferenceRole.FOUNDATIONAL
        
        # Check if it identifies research gaps
        if research_gap:
            return ReferenceRole.GAP_IDENTIFYING
        
        # Check if it's primarily methodological
        if len(methodological_approaches) > 5:
            return ReferenceRole.METHODOLOGICAL
        
        # Check for contrasting perspectives
        if reference.title and any(
            word in reference.title.lower() 
            for word in ['alternative', 'different', 'novel', 'contrast']
        ):
            return ReferenceRole.CONTRASTING
        
        # Default to supporting evidence
        return ReferenceRole.SUPPORTING
    
    def _create_reference_context(
        self, 
        analysis: ReferenceAnalysis, 
        study_spec: StudySpecification
    ) -> ReferenceContext:
        """Create reference context from analysis."""
        
        # Format citation
        citation_format = self.apa_formatter.format_reference(analysis.reference)
        
        # Create citation context based on role
        context_templates = {
            ReferenceRole.FOUNDATIONAL: "Foundational research has established that",
            ReferenceRole.SUPPORTING: "Previous studies have demonstrated that", 
            ReferenceRole.CONTRASTING: "However, alternative perspectives suggest that",
            ReferenceRole.GAP_IDENTIFYING: "Despite advances, researchers have noted that",
            ReferenceRole.METHODOLOGICAL: "Methodological approaches have shown that"
        }
        
        citation_context = context_templates.get(
            analysis.recommended_role, 
            "Research has shown that"
        )
        
        return ReferenceContext(
            reference=analysis.reference,
            role=analysis.recommended_role,
            key_concepts=analysis.key_concepts,
            key_findings=analysis.key_findings,
            methodological_details=analysis.methodological_approaches,
            limitations_identified=analysis.limitations_identified,
            citation_context=citation_context,
            paragraph_target=0,  # Will be assigned later
            citation_format=citation_format
        )
    
    def _balance_reference_roles(self, contexts: List[ReferenceContext]) -> List[ReferenceContext]:
        """Ensure balanced distribution of reference roles."""
        # Count current distribution
        role_counts = {}
        for context in contexts:
            role_counts[context.role] = role_counts.get(context.role, 0) + 1
        
        # Adjust if needed (simple strategy)
        total_refs = len(contexts)
        if total_refs > 0:
            # Ensure at least one foundational reference
            if role_counts.get(ReferenceRole.FOUNDATIONAL, 0) == 0 and contexts:
                contexts[0].role = ReferenceRole.FOUNDATIONAL
        
        return contexts
    
    def _assign_paragraph_targets(self, contexts: List[ReferenceContext]) -> List[ReferenceContext]:
        """Assign paragraph targets based on reference roles."""
        role_to_paragraph = {
            ReferenceRole.FOUNDATIONAL: 1,
            ReferenceRole.SUPPORTING: 2,
            ReferenceRole.CONTRASTING: 2,
            ReferenceRole.GAP_IDENTIFYING: 3,
            ReferenceRole.METHODOLOGICAL: 4
        }
        
        for context in contexts:
            context.paragraph_target = role_to_paragraph.get(context.role, 2)
        
        return contexts
    
    def _find_processed_document(
        self, 
        reference: Reference, 
        processed_documents: Optional[List[ProcessedDocument]]
    ) -> Optional[ProcessedDocument]:
        """Find processed document corresponding to reference."""
        if not processed_documents:
            return None
        
        # Try to match by title or DOI
        for doc in processed_documents:
            if reference.title and doc.title and reference.title.lower() in doc.title.lower():
                return doc
            if reference.doi and doc.metadata and reference.doi in str(doc.metadata):
                return doc
        
        return None
    
    def _create_minimal_analysis(self, reference: Reference) -> ReferenceAnalysis:
        """Create minimal analysis when full analysis fails."""
        return ReferenceAnalysis(
            reference=reference,
            key_concepts=[],
            key_findings=[],
            methodological_approaches=[],
            limitations_identified=[],
            research_gap_identified=None,
            theoretical_framework=None,
            clinical_significance=None,
            relevance_score=0.5,  # Default medium relevance
            recommended_role=ReferenceRole.SUPPORTING,
            confidence_score=0.3  # Low confidence
        )
    
    def _extract_study_concepts(self, study_spec: StudySpecification) -> List[str]:
        """Extract key concepts from study specification."""
        concepts = []
        
        text = f"{study_spec.study_title} {study_spec.primary_research_question} {study_spec.primary_hypothesis}"
        text += f" {study_spec.methodology_summary} {study_spec.clinical_significance}"
        
        for concept in self.neuroscience_concepts:
            if concept.lower() in text.lower():
                concepts.append(concept)
        
        return concepts
    
    def _calculate_analysis_confidence(
        self, 
        text: str, 
        key_concepts: List[str], 
        key_findings: List[str]
    ) -> float:
        """Calculate confidence in the reference analysis."""
        confidence = 0.0
        
        # Text availability
        if text and len(text) > 100:
            confidence += 0.4
        elif text:
            confidence += 0.2
        
        # Concept extraction success
        if key_concepts:
            confidence += 0.3
        
        # Finding extraction success
        if key_findings:
            confidence += 0.3
        
        return min(confidence, 1.0)
    
    def _log_role_distribution(self, contexts: List[ReferenceContext]):
        """Log distribution of reference roles."""
        role_counts = {}
        for context in contexts:
            role_counts[context.role] = role_counts.get(context.role, 0) + 1
        
        logger.info("Reference role distribution:")
        for role, count in role_counts.items():
            logger.info(f"  {role.value}: {count}")
    
    def _load_neuroscience_concepts(self) -> List[str]:
        """Load neuroscience concepts for concept extraction."""
        return [
            "neuroplasticity", "synaptic", "neurotransmitter", "cortical", "hippocampus",
            "amygdala", "dopamine", "serotonin", "GABA", "glutamate", "neural network",
            "action potential", "dendrite", "axon", "myelin", "glia", "astrocyte",
            "microglia", "blood-brain barrier", "cerebrospinal fluid", "EEG", "fMRI",
            "PET scan", "DTI", "optogenetics", "electrophysiology", "calcium imaging",
            "patch clamp", "two-photon", "confocal", "immunohistochemistry", "Western blot",
            "PCR", "qPCR", "RNA-seq", "proteomics", "metabolomics", "connectome",
            "default mode network", "attention", "memory", "working memory", "episodic memory",
            "semantic memory", "procedural memory", "cognition", "consciousness",
            "sleep", "circadian", "stroke", "epilepsy", "Alzheimer", "Parkinson",
            "depression", "anxiety", "schizophrenia", "autism", "ADHD", "traumatic brain injury"
        ]
    
    def _load_methodology_keywords(self) -> List[str]:
        """Load methodology keywords for methodological approach extraction."""
        return [
            "fMRI", "functional magnetic resonance imaging", "EEG", "electroencephalography",
            "MEG", "magnetoencephalography", "PET", "positron emission tomography",
            "DTI", "diffusion tensor imaging", "optogenetics", "electrophysiology",
            "calcium imaging", "two-photon microscopy", "confocal microscopy",
            "patch clamp", "whole-cell recording", "extracellular recording",
            "immunohistochemistry", "immunofluorescence", "Western blot",
            "RT-PCR", "qPCR", "RNA sequencing", "single-cell RNA-seq",
            "proteomics", "metabolomics", "ChIP-seq", "ATAC-seq",
            "behavioral testing", "cognitive assessment", "neuropsychological testing",
            "lesion study", "pharmacological intervention", "genetic manipulation",
            "transgenic mice", "knockout mice", "viral vectors", "CRISPR",
            "machine learning", "artificial neural networks", "computational modeling",
            "statistical analysis", "meta-analysis", "systematic review"
        ]
    
    def _get_domain_specific_keywords(self, domain: str) -> List[str]:
        """Get keywords specific to research domain."""
        domain_keywords = {
            'neurosurgery': [
                'surgical', 'operative', 'resection', 'biopsy', 'craniotomy',
                'stereotactic', 'navigation', 'intraoperative', 'tumor', 'glioma',
                'meningioma', 'metastasis', 'aneurysm', 'AVM', 'deep brain stimulation'
            ],
            'cognitive_neuroscience': [
                'cognitive', 'attention', 'memory', 'executive function', 'language',
                'decision making', 'reward processing', 'social cognition', 'emotion',
                'perception', 'consciousness', 'learning'
            ],
            'neuroimaging': [
                'neuroimaging', 'brain imaging', 'structural MRI', 'functional MRI',
                'diffusion MRI', 'perfusion', 'spectroscopy', 'connectivity',
                'network analysis', 'voxel-based morphometry', 'surface-based analysis'
            ]
        }
        
        return domain_keywords.get(domain, [])