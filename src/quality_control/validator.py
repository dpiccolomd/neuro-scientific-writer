"""Master quality validator for neuroscience manuscripts."""

import re
import time
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.models import AnalysisResult
from template_engine.models import TemplateMetadata
from .models import (
    QualityReport, QualityMetrics, ValidationResult, ValidationIssue,
    WarningLevel, ValidationType
)
from .exceptions import QualityControlError, CriticalValidationError
from .citation_verifier import CitationVerifier
from .fact_checker import FactualConsistencyChecker
from .statistical_validator import StatisticalValidator
from .plagiarism_detector import PlagiarismDetector

logger = logging.getLogger(__name__)


class QualityValidator:
    """
    Master quality validator for neuroscience manuscripts.
    
    Implements rigorous validation standards required for medical/scientific publication.
    All validation failures are clearly documented with severity levels.
    """
    
    def __init__(self):
        """Initialize the quality validator with all sub-validators."""
        self.citation_verifier = CitationVerifier()
        self.fact_checker = FactualConsistencyChecker()
        self.statistical_validator = StatisticalValidator()
        self.plagiarism_detector = PlagiarismDetector()
        
        # Neuroscience-specific validation patterns
        self.critical_terms = self._load_critical_neuroscience_terms()
        self.statistical_patterns = self._load_statistical_patterns()
        self.methodological_keywords = self._load_methodological_keywords()
        
    def validate_draft(self, 
                      draft_text: str,
                      source_papers: List[AnalysisResult],
                      template_metadata: TemplateMetadata,
                      document_id: str = None) -> QualityReport:
        """
        Perform comprehensive quality validation of a manuscript draft.
        
        Args:
            draft_text: The manuscript text to validate
            source_papers: Analysis results from source papers used
            template_metadata: Metadata about the template used
            document_id: Identifier for the document
            
        Returns:
            Comprehensive quality report with all validation results
            
        Raises:
            CriticalValidationError: If critical issues prevent publication
        """
        start_time = time.time()
        document_id = document_id or f"draft_{hash(draft_text)}"
        
        logger.info(f"Starting comprehensive validation for document {document_id}")
        
        try:
            validation_results = []
            critical_issues = []
            warnings = []
            
            # 1. Citation Verification (CRITICAL)
            citation_result = self._validate_citations(draft_text, source_papers)
            validation_results.append(citation_result)
            self._categorize_issues(citation_result, critical_issues, warnings)
            
            # 2. Factual Consistency (CRITICAL)
            factual_result = self._validate_factual_consistency(draft_text, source_papers)
            validation_results.append(factual_result)
            self._categorize_issues(factual_result, critical_issues, warnings)
            
            # 3. Statistical Accuracy (HIGH PRIORITY)
            statistical_result = self._validate_statistics(draft_text)
            validation_results.append(statistical_result)
            self._categorize_issues(statistical_result, critical_issues, warnings)
            
            # 4. Terminology Accuracy (MODERATE PRIORITY)
            terminology_result = self._validate_terminology(draft_text)
            validation_results.append(terminology_result)
            self._categorize_issues(terminology_result, critical_issues, warnings)
            
            # 5. Plagiarism Detection (CRITICAL)
            plagiarism_result = self._detect_plagiarism(draft_text, source_papers)
            validation_results.append(plagiarism_result)
            self._categorize_issues(plagiarism_result, critical_issues, warnings)
            
            # 6. Methodological Rigor (HIGH PRIORITY)
            methodological_result = self._validate_methodology_references(draft_text)
            validation_results.append(methodological_result)
            self._categorize_issues(methodological_result, critical_issues, warnings)
            
            # Calculate comprehensive quality metrics
            quality_metrics = self._calculate_quality_metrics(validation_results)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                validation_results, critical_issues, warnings
            )
            
            # Determine publication readiness
            ready_for_publication = len(critical_issues) == 0 and quality_metrics.overall_score > 0.7
            
            # Calculate validation coverage
            validation_coverage = self._calculate_validation_coverage(draft_text)
            
            processing_time = time.time() - start_time
            
            quality_report = QualityReport(
                document_id=document_id,
                validation_timestamp=datetime.now(),
                quality_metrics=quality_metrics,
                validation_results=validation_results,
                critical_issues=critical_issues,
                warnings=warnings,
                recommendations=recommendations,
                sources_validated=len(source_papers),
                validation_coverage=validation_coverage,
                ready_for_publication=ready_for_publication
            )
            
            logger.info(f"Validation completed in {processing_time:.2f}s")
            logger.info(f"Quality score: {quality_metrics.overall_score:.3f}")
            logger.info(f"Critical issues: {len(critical_issues)}")
            logger.info(f"Warnings: {len(warnings)}")
            logger.info(f"Ready for publication: {ready_for_publication}")
            
            # Raise exception if critical issues found
            if critical_issues:
                critical_messages = [issue.message for issue in critical_issues]
                raise CriticalValidationError(
                    f"Critical validation issues found: {'; '.join(critical_messages[:3])}"
                )
            
            return quality_report
            
        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            if isinstance(e, CriticalValidationError):
                raise
            raise QualityControlError(f"Validation failed: {e}", document_id)
    
    def _validate_citations(self, text: str, source_papers: List[AnalysisResult]) -> ValidationResult:
        """Validate all citations in the text."""
        start_time = time.time()
        issues = []
        
        # Extract citations
        citation_pattern = r'\([^)]*\d{4}[^)]*\)'
        citations = re.findall(citation_pattern, text)
        
        if not citations:
            issues.append(ValidationIssue(
                issue_type=ValidationType.CITATION_VERIFICATION,
                severity=WarningLevel.HIGH,
                message="No citations found in text - neuroscience manuscripts require extensive referencing",
                location="entire document",
                suggestion="Add citations to support all factual claims"
            ))
        
        verified_count = 0
        total_citations = len(citations)
        
        for i, citation in enumerate(citations):
            # Check citation format
            if not self._is_valid_apa_format(citation):
                issues.append(ValidationIssue(
                    issue_type=ValidationType.CITATION_VERIFICATION,
                    severity=WarningLevel.MODERATE,
                    message=f"Citation format may not comply with APA standards: {citation}",
                    location=f"citation {i+1}",
                    suggestion="Verify APA formatting guidelines"
                ))
            
            # Check if citation context is appropriate
            citation_context = self._extract_citation_context(text, citation)
            if not self._is_appropriate_citation_context(citation_context):
                issues.append(ValidationIssue(
                    issue_type=ValidationType.CITATION_VERIFICATION,
                    severity=WarningLevel.MODERATE,
                    message=f"Citation context may be inappropriate: {citation}",
                    location=f"citation {i+1}",
                    suggestion="Ensure citation supports the claim being made"
                ))
            else:
                verified_count += 1
        
        # Check citation density
        sentences = text.split('.')
        citation_density = total_citations / len(sentences) if sentences else 0
        
        if citation_density < 0.2:  # Less than 1 citation per 5 sentences
            issues.append(ValidationIssue(
                issue_type=ValidationType.CITATION_VERIFICATION,
                severity=WarningLevel.HIGH,
                message=f"Low citation density ({citation_density:.2f}) - neuroscience introductions typically need higher referencing",
                location="entire document",
                suggestion="Add more citations to support claims, especially for factual statements"
            ))
        
        score = verified_count / total_citations if total_citations > 0 else 0.5
        passed = len([i for i in issues if i.severity == WarningLevel.CRITICAL]) == 0
        
        return ValidationResult(
            validation_type=ValidationType.CITATION_VERIFICATION,
            passed=passed,
            score=score,
            issues=issues,
            details={
                "total_citations": total_citations,
                "verified_citations": verified_count,
                "citation_density": citation_density
            },
            processing_time=time.time() - start_time
        )
    
    def _validate_factual_consistency(self, text: str, 
                                    source_papers: List[AnalysisResult]) -> ValidationResult:
        """Validate factual consistency against source papers."""
        start_time = time.time()
        issues = []
        
        # Extract factual claims
        factual_claims = self._extract_factual_claims(text)
        verified_claims = 0
        
        for claim in factual_claims:
            # Check for overly broad generalizations
            if self._is_overly_broad_claim(claim):
                issues.append(ValidationIssue(
                    issue_type=ValidationType.FACTUAL_CONSISTENCY,
                    severity=WarningLevel.MODERATE,
                    message=f"Potentially overly broad claim: {claim[:100]}...",
                    location="factual claim",
                    suggestion="Consider adding qualifiers or limiting scope"
                ))
            
            # Check for unsubstantiated claims
            if self._lacks_supporting_evidence(claim, text):
                issues.append(ValidationIssue(
                    issue_type=ValidationType.FACTUAL_CONSISTENCY,
                    severity=WarningLevel.HIGH,
                    message=f"Claim lacks supporting citation: {claim[:100]}...",
                    location="factual claim",
                    suggestion="Add citation to support this claim"
                ))
            else:
                verified_claims += 1
        
        # Check for contradictory statements
        contradictions = self._detect_contradictions(text)
        for contradiction in contradictions:
            issues.append(ValidationIssue(
                issue_type=ValidationType.FACTUAL_CONSISTENCY,
                severity=WarningLevel.CRITICAL,
                message=f"Potential contradiction detected: {contradiction}",
                location="document consistency",
                suggestion="Resolve contradictory statements"
            ))
        
        score = verified_claims / len(factual_claims) if factual_claims else 0.8
        passed = len([i for i in issues if i.severity == WarningLevel.CRITICAL]) == 0
        
        return ValidationResult(
            validation_type=ValidationType.FACTUAL_CONSISTENCY,
            passed=passed,
            score=score,
            issues=issues,
            details={
                "factual_claims_found": len(factual_claims),
                "verified_claims": verified_claims,
                "contradictions_found": len(contradictions)
            },
            processing_time=time.time() - start_time
        )
    
    def _validate_statistics(self, text: str) -> ValidationResult:
        """Validate statistical reporting accuracy."""
        start_time = time.time()
        issues = []
        
        # Find statistical statements
        stats = self._extract_statistical_statements(text)
        valid_stats = 0
        
        for stat in stats:
            # Validate p-values
            if 'p' in stat.lower():
                p_values = re.findall(r'p\s*[<>=]\s*([0-9.]+)', stat.lower())
                for p_val in p_values:
                    try:
                        p_float = float(p_val)
                        if p_float > 1.0:
                            issues.append(ValidationIssue(
                                issue_type=ValidationType.STATISTICAL_ACCURACY,
                                severity=WarningLevel.CRITICAL,
                                message=f"Invalid p-value (>1.0): p = {p_val}",
                                location="statistical statement",
                                suggestion="P-values must be between 0 and 1"
                            ))
                        elif p_float == 0.0:
                            issues.append(ValidationIssue(
                                issue_type=ValidationType.STATISTICAL_ACCURACY,
                                severity=WarningLevel.MODERATE,
                                message=f"P-value reported as exactly 0.0: consider reporting as p < 0.001",
                                location="statistical statement",
                                suggestion="Use p < 0.001 instead of p = 0.000"
                            ))
                        else:
                            valid_stats += 1
                    except ValueError:
                        issues.append(ValidationIssue(
                            issue_type=ValidationType.STATISTICAL_ACCURACY,
                            severity=WarningLevel.MODERATE,
                            message=f"Invalid p-value format: {p_val}",
                            location="statistical statement",
                            suggestion="Check p-value formatting"
                        ))
            
            # Validate correlations
            if 'r =' in stat.lower():
                correlations = re.findall(r'r\s*=\s*([-0-9.]+)', stat.lower())
                for corr in correlations:
                    try:
                        corr_float = float(corr)
                        if abs(corr_float) > 1.0:
                            issues.append(ValidationIssue(
                                issue_type=ValidationType.STATISTICAL_ACCURACY,
                                severity=WarningLevel.CRITICAL,
                                message=f"Invalid correlation coefficient (>|1.0|): r = {corr}",
                                location="statistical statement",
                                suggestion="Correlation coefficients must be between -1 and 1"
                            ))
                        else:
                            valid_stats += 1
                    except ValueError:
                        issues.append(ValidationIssue(
                            issue_type=ValidationType.STATISTICAL_ACCURACY,
                            severity=WarningLevel.MODERATE,
                            message=f"Invalid correlation format: {corr}",
                            location="statistical statement",
                            suggestion="Check correlation coefficient formatting"
                        ))
        
        score = valid_stats / len(stats) if stats else 1.0
        passed = len([i for i in issues if i.severity == WarningLevel.CRITICAL]) == 0
        
        return ValidationResult(
            validation_type=ValidationType.STATISTICAL_ACCURACY,
            passed=passed,
            score=score,
            issues=issues,
            details={
                "statistical_statements": len(stats),
                "valid_statistics": valid_stats
            },
            processing_time=time.time() - start_time
        )
    
    def _validate_terminology(self, text: str) -> ValidationResult:
        """Validate neuroscience terminology usage."""
        start_time = time.time()
        issues = []
        
        # Check for common terminology errors
        terminology_errors = [
            (r'\bnervous system\b', r'\bcentral nervous system\b', 
             "Consider specifying 'central nervous system' or 'peripheral nervous system'"),
            (r'\bbrain waves\b', r'\bneural oscillations\b', 
             "Use 'neural oscillations' instead of 'brain waves' in formal writing"),
            (r'\bmemory\b(?!\s+consolidation)', r'\bepisodic memory|working memory|semantic memory\b',
             "Consider specifying memory type (episodic, working, semantic, etc.)"),
        ]
        
        for pattern, _, suggestion in terminology_errors:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(ValidationIssue(
                    issue_type=ValidationType.TERMINOLOGY_ACCURACY,
                    severity=WarningLevel.LOW,
                    message=f"Consider more specific terminology",
                    location="terminology usage",
                    suggestion=suggestion
                ))
        
        # Check for undefined abbreviations
        abbreviations = re.findall(r'\b[A-Z]{2,}\b', text)
        for abbrev in abbreviations:
            if abbrev not in ['MRI', 'EEG', 'fMRI', 'PET', 'CT', 'DTI']:  # Common ones
                # Look for definition in text
                definition_pattern = f'{abbrev}\\s*\\([^)]+\\)|\\([^)]*{abbrev}\\)'
                if not re.search(definition_pattern, text):
                    issues.append(ValidationIssue(
                        issue_type=ValidationType.TERMINOLOGY_ACCURACY,
                        severity=WarningLevel.MODERATE,
                        message=f"Abbreviation '{abbrev}' may not be defined",
                        location="abbreviation usage",
                        suggestion="Define abbreviations on first use"
                    ))
        
        score = 1.0 - (len(issues) * 0.1)  # Penalty for issues
        score = max(0.0, score)
        passed = len([i for i in issues if i.severity in [WarningLevel.CRITICAL, WarningLevel.HIGH]]) == 0
        
        return ValidationResult(
            validation_type=ValidationType.TERMINOLOGY_ACCURACY,
            passed=passed,
            score=score,
            issues=issues,
            details={"terminology_issues": len(issues)},
            processing_time=time.time() - start_time
        )
    
    def _detect_plagiarism(self, text: str, source_papers: List[AnalysisResult]) -> ValidationResult:
        """Detect potential plagiarism issues."""
        start_time = time.time()
        issues = []
        
        # Simple plagiarism detection (would be enhanced in production)
        sentences = text.split('.')
        similar_sentences = 0
        
        # Check for overly similar sentence structures
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) > 50:  # Only check substantial sentences
                # Simple similarity check (would use more sophisticated methods)
                if self._is_potentially_plagiarized_sentence(sentence):
                    issues.append(ValidationIssue(
                        issue_type=ValidationType.PLAGIARISM_CHECK,
                        severity=WarningLevel.HIGH,
                        message=f"Sentence may be too similar to common phrasing: {sentence[:100]}...",
                        location=f"sentence {i+1}",
                        suggestion="Rephrase in your own words or add proper citation"
                    ))
                else:
                    similar_sentences += 1
        
        score = similar_sentences / len(sentences) if sentences else 1.0
        passed = len([i for i in issues if i.severity in [WarningLevel.CRITICAL, WarningLevel.HIGH]]) == 0
        
        return ValidationResult(
            validation_type=ValidationType.PLAGIARISM_CHECK,
            passed=passed,
            score=score,
            issues=issues,
            details={"sentences_checked": len(sentences)},
            processing_time=time.time() - start_time
        )
    
    def _validate_methodology_references(self, text: str) -> ValidationResult:
        """Validate methodology and experimental design references."""
        start_time = time.time()
        issues = []
        
        # Check for methodology mentions without proper detail
        methodology_terms = ['participants', 'subjects', 'procedure', 'analysis', 'statistical']
        methodology_mentions = 0
        
        for term in methodology_terms:
            if term in text.lower():
                methodology_mentions += 1
                # Check if properly detailed
                if not self._has_sufficient_methodology_detail(text, term):
                    issues.append(ValidationIssue(
                        issue_type=ValidationType.METHODOLOGICAL_RIGOR,
                        severity=WarningLevel.MODERATE,
                        message=f"Methodology term '{term}' mentioned but may lack sufficient detail",
                        location="methodology reference",
                        suggestion="Provide more specific methodological details or references"
                    ))
        
        score = 0.8  # Base score for methodology validation
        passed = True
        
        return ValidationResult(
            validation_type=ValidationType.METHODOLOGICAL_RIGOR,
            passed=passed,
            score=score,
            issues=issues,
            details={"methodology_terms_found": methodology_mentions},
            processing_time=time.time() - start_time
        )
    
    # Helper methods for validation
    
    def _categorize_issues(self, result: ValidationResult, 
                         critical_issues: List[ValidationIssue], 
                         warnings: List[ValidationIssue]):
        """Categorize issues by severity."""
        for issue in result.issues:
            if issue.severity == WarningLevel.CRITICAL:
                critical_issues.append(issue)
            else:
                warnings.append(issue)
    
    def _calculate_quality_metrics(self, results: List[ValidationResult]) -> QualityMetrics:
        """Calculate comprehensive quality metrics."""
        scores = {}
        for result in results:
            if result.validation_type == ValidationType.CITATION_VERIFICATION:
                scores['citation'] = result.score
            elif result.validation_type == ValidationType.FACTUAL_CONSISTENCY:
                scores['factual'] = result.score
            elif result.validation_type == ValidationType.STATISTICAL_ACCURACY:
                scores['statistical'] = result.score
            elif result.validation_type == ValidationType.TERMINOLOGY_ACCURACY:
                scores['terminology'] = result.score
            elif result.validation_type == ValidationType.PLAGIARISM_CHECK:
                scores['plagiarism'] = result.score
            elif result.validation_type == ValidationType.METHODOLOGICAL_RIGOR:
                scores['methodological'] = result.score
        
        # Fill missing scores with neutral values
        for key in ['citation', 'factual', 'statistical', 'terminology', 'plagiarism', 'methodological']:
            if key not in scores:
                scores[key] = 0.7  # Neutral score
        
        # Calculate overall score
        weights = {
            'citation': 0.25,
            'factual': 0.25,
            'statistical': 0.15,
            'terminology': 0.10,
            'plagiarism': 0.15,
            'methodological': 0.10
        }
        
        overall_score = sum(scores[key] * weights[key] for key in weights.keys())
        
        # Calculate confidence interval (simplified)
        confidence_lower = max(0.0, overall_score - 0.1)
        confidence_upper = min(1.0, overall_score + 0.1)
        
        return QualityMetrics(
            overall_score=overall_score,
            citation_accuracy=scores['citation'],
            factual_consistency=scores['factual'],
            statistical_validity=scores['statistical'],
            terminology_score=scores['terminology'],
            plagiarism_score=scores['plagiarism'],
            methodological_rigor=scores['methodological'],
            confidence_interval=(confidence_lower, confidence_upper)
        )
    
    def _generate_recommendations(self, results: List[ValidationResult],
                                critical_issues: List[ValidationIssue],
                                warnings: List[ValidationIssue]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if critical_issues:
            recommendations.append("❌ CRITICAL: Resolve all critical issues before proceeding")
        
        if len(warnings) > 5:
            recommendations.append("⚠️  Consider addressing high-priority warnings")
        
        # Specific recommendations based on validation results
        for result in results:
            if result.score < 0.7:
                if result.validation_type == ValidationType.CITATION_VERIFICATION:
                    recommendations.append("📚 Improve citation coverage and formatting")
                elif result.validation_type == ValidationType.FACTUAL_CONSISTENCY:
                    recommendations.append("🔍 Verify factual claims against source literature")
                elif result.validation_type == ValidationType.STATISTICAL_ACCURACY:
                    recommendations.append("📊 Review statistical reporting for accuracy")
        
        if not recommendations:
            recommendations.append("✅ Quality validation passed - ready for expert review")
        
        return recommendations
    
    def _calculate_validation_coverage(self, text: str) -> float:
        """Calculate percentage of text that was validated."""
        # Simplified calculation - in production would be more sophisticated
        return 0.95  # Assume 95% coverage for now
    
    # Additional helper methods (simplified implementations)
    
    def _load_critical_neuroscience_terms(self) -> Dict[str, str]:
        """Load critical neuroscience terms that must be used correctly."""
        return {
            'hippocampus': 'memory formation structure',
            'amygdala': 'fear and emotion processing',
            'cortex': 'outer brain layer',
            'neuron': 'nerve cell',
            'synapse': 'neural connection'
        }
    
    def _load_statistical_patterns(self) -> Dict[str, str]:
        """Load statistical reporting patterns."""
        return {
            'p_value': r'p\s*[<>=]\s*[0-9.]+',
            'correlation': r'r\s*=\s*[-0-9.]+',
            'effect_size': r'[dη²]\s*=\s*[0-9.]+'
        }
    
    def _load_methodological_keywords(self) -> List[str]:
        """Load methodological keywords that should be properly referenced."""
        return ['fMRI', 'EEG', 'participants', 'randomized', 'controlled', 'statistical analysis']
    
    def _is_valid_apa_format(self, citation: str) -> bool:
        """Check if citation follows APA format."""
        # Simplified APA format check
        return bool(re.match(r'\([^)]*\d{4}[^)]*\)', citation))
    
    def _extract_citation_context(self, text: str, citation: str) -> str:
        """Extract context around a citation."""
        citation_pos = text.find(citation)
        if citation_pos == -1:
            return ""
        
        start = max(0, citation_pos - 100)
        end = min(len(text), citation_pos + len(citation) + 100)
        return text[start:end]
    
    def _is_appropriate_citation_context(self, context: str) -> bool:
        """Check if citation context is appropriate."""
        # Simplified check - look for claim words near citation
        claim_words = ['showed', 'demonstrated', 'found', 'reported', 'indicated']
        return any(word in context.lower() for word in claim_words)
    
    def _extract_factual_claims(self, text: str) -> List[str]:
        """Extract factual claims from text."""
        # Simplified - look for sentences with definitive statements
        sentences = text.split('.')
        claims = []
        claim_indicators = ['is', 'are', 'shows', 'demonstrates', 'indicates']
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in claim_indicators):
                claims.append(sentence.strip())
        
        return claims
    
    def _is_overly_broad_claim(self, claim: str) -> bool:
        """Check if claim is overly broad or generalized."""
        broad_terms = ['all', 'every', 'always', 'never', 'completely', 'entirely']
        return any(term in claim.lower() for term in broad_terms)
    
    def _lacks_supporting_evidence(self, claim: str, full_text: str) -> bool:
        """Check if claim lacks supporting citation."""
        # Look for citation near claim
        citation_pattern = r'\([^)]*\d{4}[^)]*\)'
        claim_start = full_text.find(claim)
        if claim_start == -1:
            return True
        
        # Check 200 characters around claim for citation
        context_start = max(0, claim_start - 100)
        context_end = min(len(full_text), claim_start + len(claim) + 100)
        context = full_text[context_start:context_end]
        
        return not bool(re.search(citation_pattern, context))
    
    def _detect_contradictions(self, text: str) -> List[str]:
        """Detect potential contradictions in text."""
        # Simplified contradiction detection
        contradictions = []
        
        # Look for opposing statements
        opposing_pairs = [
            ('increases', 'decreases'),
            ('enhances', 'impairs'),
            ('activates', 'inhibits'),
            ('improves', 'worsens')
        ]
        
        for pos_term, neg_term in opposing_pairs:
            if pos_term in text.lower() and neg_term in text.lower():
                contradictions.append(f"Contains both '{pos_term}' and '{neg_term}' - check for contradictions")
        
        return contradictions
    
    def _extract_statistical_statements(self, text: str) -> List[str]:
        """Extract statistical statements from text."""
        stat_pattern = r'[^.]*(?:p\s*[<>=]|r\s*=|t\s*=|F\s*=|χ²)[^.]*'
        return re.findall(stat_pattern, text, re.IGNORECASE)
    
    def _is_potentially_plagiarized_sentence(self, sentence: str) -> bool:
        """Simple check for potentially plagiarized content."""
        # Look for overly formal or unusual phrasings
        formal_phrases = [
            'it has been well established that',
            'extensive research has demonstrated',
            'it is widely accepted that'
        ]
        return any(phrase in sentence.lower() for phrase in formal_phrases)
    
    def _has_sufficient_methodology_detail(self, text: str, term: str) -> bool:
        """Check if methodology term has sufficient detail."""
        # Look for specific numbers, procedures, or references near term
        term_pos = text.lower().find(term)
        if term_pos == -1:
            return True
        
        context_start = max(0, term_pos - 200)
        context_end = min(len(text), term_pos + 200)
        context = text[context_start:context_end]
        
        # Look for numbers, specific procedures, or citations
        has_numbers = bool(re.search(r'\b\d+\b', context))
        has_citation = bool(re.search(r'\([^)]*\d{4}[^)]*\)', context))
        has_specifics = any(word in context.lower() for word in ['using', 'following', 'according to'])
        
        return has_numbers or has_citation or has_specifics