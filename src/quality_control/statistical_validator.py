"""Statistical validation for neuroscience manuscripts."""

import re
import logging
from typing import List, Dict, Optional, Tuple
from .models import StatisticalCheck, ValidationIssue, WarningLevel, ValidationType

logger = logging.getLogger(__name__)


class StatisticalValidator:
    """Validates statistical reporting in neuroscience manuscripts."""
    
    def __init__(self):
        """Initialize statistical validator."""
        self.statistical_patterns = self._load_statistical_patterns()
        self.acceptable_ranges = self._load_acceptable_ranges()
    
    def validate_statistics(self, text: str) -> List[StatisticalCheck]:
        """
        Validate all statistical reports in the text.
        
        Args:
            text: Text containing statistical reports
            
        Returns:
            List of statistical validation results
        """
        checks = []
        
        # Find all statistical statements
        for stat_type, pattern in self.statistical_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                check = self._validate_statistical_statement(
                    match.group(), stat_type, match.span()
                )
                checks.append(check)
        
        return checks
    
    def _load_statistical_patterns(self) -> Dict[str, str]:
        """Load patterns for different statistical reports."""
        return {
            'p_value': r'p\s*[<>=]\s*([0-9.]+)',
            'correlation': r'r\s*=\s*([-0-9.]+)',
            'cohen_d': r"Cohen'?s?\s*d\s*=\s*([0-9.]+)",
            'eta_squared': r'η²\s*=\s*([0-9.]+)',
            't_test': r't\s*\(\s*(\d+)\s*\)\s*=\s*([-0-9.]+)',
            'f_test': r'F\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*([0-9.]+)',
            'chi_square': r'χ²\s*\(\s*(\d+)\s*\)\s*=\s*([0-9.]+)',
            'confidence_interval': r'95%\s*CI\s*[\[\(]\s*([-0-9.]+)\s*,\s*([-0-9.]+)\s*[\]\)]'
        }
    
    def _load_acceptable_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Load acceptable ranges for different statistics."""
        return {
            'p_value': (0.0, 1.0),
            'correlation': (-1.0, 1.0),
            'cohen_d': (-10.0, 10.0),  # Generous range for effect sizes
            'eta_squared': (0.0, 1.0),
            't_test': (-100.0, 100.0),  # Very generous range
            'f_test': (0.0, 1000.0),  # F-statistics are non-negative
            'chi_square': (0.0, 1000.0)  # Chi-square is non-negative
        }
    
    def _validate_statistical_statement(self, statement: str, 
                                      stat_type: str, 
                                      location: Tuple[int, int]) -> StatisticalCheck:
        """Validate a single statistical statement."""
        issues = []
        
        # Extract the numeric value(s)
        if stat_type == 'p_value':
            value = self._extract_p_value(statement)
            issues.extend(self._validate_p_value(value, statement))
        elif stat_type == 'correlation':
            value = self._extract_correlation(statement)
            issues.extend(self._validate_correlation(value, statement))
        elif stat_type == 'cohen_d':
            value = self._extract_effect_size(statement)
            issues.extend(self._validate_effect_size(value, statement, 'Cohen\'s d'))
        elif stat_type == 'eta_squared':
            value = self._extract_eta_squared(statement)
            issues.extend(self._validate_eta_squared(value, statement))
        else:
            value = None
        
        # General format validation
        format_correct = self._check_format(statement, stat_type)
        if not format_correct:
            issues.append(ValidationIssue(
                issue_type=ValidationType.STATISTICAL_ACCURACY,
                severity=WarningLevel.MODERATE,
                message=f"Statistical format may be non-standard: {statement}",
                location=f"position {location[0]}-{location[1]}",
                suggestion="Check APA guidelines for statistical reporting"
            ))
        
        # Context appropriateness
        context_appropriate = self._check_statistical_context(statement)
        if not context_appropriate:
            issues.append(ValidationIssue(
                issue_type=ValidationType.STATISTICAL_ACCURACY,
                severity=WarningLevel.MODERATE,
                message=f"Statistical statement may lack context: {statement}",
                location=f"position {location[0]}-{location[1]}",
                suggestion="Provide context for statistical results (e.g., what was tested)"
            ))
        
        return StatisticalCheck(
            statistic_text=statement,
            statistic_type=stat_type,
            value_reported=value,
            value_range_valid=self._is_value_in_range(value, stat_type),
            format_correct=format_correct,
            context_appropriate=context_appropriate,
            issues=issues
        )
    
    def _extract_p_value(self, statement: str) -> Optional[float]:
        """Extract p-value from statement."""
        match = re.search(r'p\s*[<>=]\s*([0-9.]+)', statement, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None
    
    def _validate_p_value(self, value: Optional[float], statement: str) -> List[ValidationIssue]:
        """Validate p-value specific rules."""
        issues = []
        
        if value is None:
            issues.append(ValidationIssue(
                issue_type=ValidationType.STATISTICAL_ACCURACY,
                severity=WarningLevel.MODERATE,
                message="Could not parse p-value",
                location="p-value",
                suggestion="Check p-value format"
            ))
            return issues
        
        if value > 1.0:
            issues.append(ValidationIssue(
                issue_type=ValidationType.STATISTICAL_ACCURACY,
                severity=WarningLevel.CRITICAL,
                message=f"P-value cannot be greater than 1.0: {value}",
                location="p-value",
                suggestion="P-values must be between 0 and 1"
            ))
        
        if value < 0.0:
            issues.append(ValidationIssue(
                issue_type=ValidationType.STATISTICAL_ACCURACY,
                severity=WarningLevel.CRITICAL,
                message=f"P-value cannot be negative: {value}",
                location="p-value",
                suggestion="P-values must be non-negative"
            ))
        
        if value == 0.0:
            issues.append(ValidationIssue(
                issue_type=ValidationType.STATISTICAL_ACCURACY,
                severity=WarningLevel.MODERATE,
                message="P-value reported as exactly 0.0",
                location="p-value",
                suggestion="Consider reporting as p < 0.001 instead of p = 0.000"
            ))
        
        # Check for appropriate precision
        if '< 0.05' in statement and value >= 0.05:
            issues.append(ValidationIssue(
                issue_type=ValidationType.STATISTICAL_ACCURACY,
                severity=WarningLevel.HIGH,
                message=f"P-value {value} reported as < 0.05",
                location="p-value",
                suggestion="Check p-value threshold reporting"
            ))
        
        return issues
    
    def _extract_correlation(self, statement: str) -> Optional[float]:
        """Extract correlation coefficient from statement."""
        match = re.search(r'r\s*=\s*([-0-9.]+)', statement, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None
    
    def _validate_correlation(self, value: Optional[float], statement: str) -> List[ValidationIssue]:
        """Validate correlation coefficient."""
        issues = []
        
        if value is None:
            return issues
        
        if abs(value) > 1.0:
            issues.append(ValidationIssue(
                issue_type=ValidationType.STATISTICAL_ACCURACY,
                severity=WarningLevel.CRITICAL,
                message=f"Correlation coefficient cannot exceed |1.0|: r = {value}",
                location="correlation",
                suggestion="Correlation coefficients must be between -1 and 1"
            ))
        
        # Interpretation warnings
        if abs(value) > 0.9:
            issues.append(ValidationIssue(
                issue_type=ValidationType.STATISTICAL_ACCURACY,
                severity=WarningLevel.LOW,
                message=f"Very high correlation (r = {value}) - verify this is correct",
                location="correlation",
                suggestion="Double-check correlation calculation for very high values"
            ))
        
        return issues
    
    def _extract_effect_size(self, statement: str) -> Optional[float]:
        """Extract effect size from statement."""
        match = re.search(r"Cohen'?s?\s*d\s*=\s*([0-9.]+)", statement, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None
    
    def _validate_effect_size(self, value: Optional[float], statement: str, 
                            effect_type: str) -> List[ValidationIssue]:
        """Validate effect size reporting."""
        issues = []
        
        if value is None:
            return issues
        
        # Cohen's d interpretation guidelines
        if 'cohen' in effect_type.lower():
            if value > 5.0:
                issues.append(ValidationIssue(
                    issue_type=ValidationType.STATISTICAL_ACCURACY,
                    severity=WarningLevel.MODERATE,
                    message=f"Very large effect size (d = {value}) - verify calculation",
                    location="effect size",
                    suggestion="Effect sizes > 2.0 are extremely large - double-check calculation"
                ))
            elif value < 0.0:
                issues.append(ValidationIssue(
                    issue_type=ValidationType.STATISTICAL_ACCURACY,
                    severity=WarningLevel.MODERATE,
                    message=f"Negative effect size - consider reporting absolute value or clarify direction",
                    location="effect size",
                    suggestion="Clarify the direction and meaning of negative effect sizes"
                ))
        
        return issues
    
    def _extract_eta_squared(self, statement: str) -> Optional[float]:
        """Extract eta squared from statement."""
        match = re.search(r'η²\s*=\s*([0-9.]+)', statement, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None
    
    def _validate_eta_squared(self, value: Optional[float], statement: str) -> List[ValidationIssue]:
        """Validate eta squared reporting."""
        issues = []
        
        if value is None:
            return issues
        
        if value > 1.0:
            issues.append(ValidationIssue(
                issue_type=ValidationType.STATISTICAL_ACCURACY,
                severity=WarningLevel.CRITICAL,
                message=f"Eta squared cannot exceed 1.0: η² = {value}",
                location="eta squared",
                suggestion="Eta squared represents proportion of variance (0-1)"
            ))
        
        if value < 0.0:
            issues.append(ValidationIssue(
                issue_type=ValidationType.STATISTICAL_ACCURACY,
                severity=WarningLevel.CRITICAL,
                message=f"Eta squared cannot be negative: η² = {value}",
                location="eta squared",
                suggestion="Eta squared must be non-negative"
            ))
        
        return issues
    
    def _check_format(self, statement: str, stat_type: str) -> bool:
        """Check if statistical statement follows standard format."""
        # Simplified format checking
        if stat_type == 'p_value':
            return bool(re.search(r'p\s*[<>=]\s*[0-9.]+', statement, re.IGNORECASE))
        elif stat_type == 'correlation':
            return bool(re.search(r'r\s*=\s*[-0-9.]+', statement, re.IGNORECASE))
        else:
            return True  # Default to true for other types
    
    def _check_statistical_context(self, statement: str) -> bool:
        """Check if statistical statement has appropriate context."""
        # Look for context indicators around statistical statement
        context_indicators = [
            'significant', 'difference', 'correlation', 'relationship',
            'effect', 'analysis', 'test', 'comparison'
        ]
        
        return any(indicator in statement.lower() for indicator in context_indicators)
    
    def _is_value_in_range(self, value: Optional[float], stat_type: str) -> bool:
        """Check if value is within acceptable range."""
        if value is None or stat_type not in self.acceptable_ranges:
            return True  # Can't validate unknown values
        
        min_val, max_val = self.acceptable_ranges[stat_type]
        return min_val <= value <= max_val