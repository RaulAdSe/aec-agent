"""
JSON Validator for AEC Compliance Agent.

This module provides validation functionality for JSON files containing building data
extracted from CAD/Revit files. It validates against Pydantic schemas and provides
detailed error messages for debugging and quality assurance.

Features:
- Validation against Project schema
- Detailed error reporting with line numbers and field paths
- CLI interface for batch validation
- Summary statistics for validation results
- Export validation reports to various formats
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import argparse
import sys
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Pydantic imports
from pydantic import ValidationError

# Add parent directory to path for schema imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import (
    Project, ProjectMetadata, Room, Door, Wall, FireEquipment, 
    Sector, Level, EvacuationRoute, validate_project_json
)


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in the JSON data."""
    level: ValidationLevel
    field_path: str
    message: str
    line_number: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validating a JSON file."""
    file_path: Path
    is_valid: bool
    project: Optional[Project] = None
    issues: List[ValidationIssue] = None
    validation_time: float = 0.0
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []


class JSONValidator:
    """
    Validates JSON files against AEC building data schemas.
    
    This class provides comprehensive validation of building data JSON files,
    including structure validation, data consistency checks, and building
    code compliance validation.
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the JSON validator.
        
        Args:
            strict_mode: If True, treats warnings as errors
        """
        self.strict_mode = strict_mode
        self.logger = logging.getLogger(__name__)
    
    def validate_file(self, file_path: Path) -> ValidationResult:
        """
        Validate a single JSON file.
        
        Args:
            file_path: Path to the JSON file to validate
            
        Returns:
            ValidationResult with validation details
        """
        start_time = datetime.now()
        
        try:
            # Check if file exists
            if not file_path.exists():
                return ValidationResult(
                    file_path=file_path,
                    is_valid=False,
                    issues=[ValidationIssue(
                        level=ValidationLevel.ERROR,
                        field_path="file",
                        message=f"File not found: {file_path}"
                    )]
                )
            
            # Load JSON data
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Validate against schema
            return self._validate_project_data(file_path, json_data, start_time)
            
        except json.JSONDecodeError as e:
            return ValidationResult(
                file_path=file_path,
                is_valid=False,
                issues=[ValidationIssue(
                    level=ValidationLevel.ERROR,
                    field_path="json",
                    message=f"Invalid JSON format: {str(e)}",
                    line_number=getattr(e, 'lineno', None)
                )],
                validation_time=(datetime.now() - start_time).total_seconds()
            )
        
        except Exception as e:
            return ValidationResult(
                file_path=file_path,
                is_valid=False,
                issues=[ValidationIssue(
                    level=ValidationLevel.ERROR,
                    field_path="validation",
                    message=f"Validation error: {str(e)}"
                )],
                validation_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _validate_project_data(self, file_path: Path, json_data: Dict[str, Any], 
                              start_time: datetime) -> ValidationResult:
        """
        Validate project data against Pydantic schema.
        
        Args:
            file_path: Path to the file being validated
            json_data: Loaded JSON data
            start_time: Validation start time
            
        Returns:
            ValidationResult with validation details
        """
        issues = []
        project = None
        
        try:
            # Validate against Project schema
            project = validate_project_json(json_data)
            
            # Additional business logic validations
            issues.extend(self._validate_business_rules(project))
            
            # Check for warnings
            issues.extend(self._check_data_quality(project))
            
            # Determine if validation passed
            error_count = sum(1 for issue in issues if issue.level == ValidationLevel.ERROR)
            warning_count = sum(1 for issue in issues if issue.level == ValidationLevel.WARNING)
            
            is_valid = error_count == 0 and (not self.strict_mode or warning_count == 0)
            
            return ValidationResult(
                file_path=file_path,
                is_valid=is_valid,
                project=project,
                issues=issues,
                validation_time=(datetime.now() - start_time).total_seconds()
            )
            
        except ValidationError as e:
            # Convert Pydantic validation errors to our format
            for error in e.errors():
                field_path = ".".join(str(loc) for loc in error['loc'])
                message = error['msg']
                
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    field_path=field_path,
                    message=message,
                    suggestion=self._get_error_suggestion(error)
                ))
            
            return ValidationResult(
                file_path=file_path,
                is_valid=False,
                project=None,
                issues=issues,
                validation_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _validate_business_rules(self, project: Project) -> List[ValidationIssue]:
        """
        Validate business rules and building code compliance.
        
        Args:
            project: Validated project data
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Check emergency exits
        egress_doors = [door for door in project.doors if door.is_egress]
        if not egress_doors:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                field_path="doors",
                message="No emergency exit doors found",
                suggestion="Mark at least one door as is_egress=True"
            ))
        
        # Check minimum door widths for egress
        for door in egress_doors:
            if door.width < 0.8:
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    field_path=f"doors.{door.id}.width",
                    message=f"Emergency exit door {door.id} width ({door.width}m) below minimum (0.8m)",
                    suggestion="Increase door width to at least 0.8m for emergency exits"
                ))
        
        # Check fire equipment coverage
        extinguishers = [eq for eq in project.fire_equipment if eq.equipment_type == "extinguisher"]
        if len(extinguishers) == 0:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                field_path="fire_equipment",
                message="No fire extinguishers found",
                suggestion="Add fire extinguishers according to building regulations"
            ))
        
        # Check room areas
        for room in project.rooms:
            if room.area and room.area <= 0:
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    field_path=f"rooms.{room.id}.area",
                    message=f"Room {room.id} has invalid area: {room.area}",
                    suggestion="Area must be greater than 0"
                ))
        
        # Check evacuation distances
        for room in project.rooms:
            if room.use_type in ["office", "retail", "commercial"] and not self._has_nearby_exit(room, project):
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    field_path=f"rooms.{room.id}",
                    message=f"Room {room.id} may be too far from emergency exits",
                    suggestion="Check evacuation distances comply with building codes"
                ))
        
        # Check fire sector sizes
        for sector in project.sectors:
            if sector.area and sector.area > 2500:  # CTE-SI typical limit
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    field_path=f"sectors.{sector.id}.area",
                    message=f"Fire sector {sector.id} area ({sector.area}m²) exceeds typical limit",
                    suggestion="Consider subdividing large fire sectors"
                ))
        
        return issues
    
    def _check_data_quality(self, project: Project) -> List[ValidationIssue]:
        """
        Check data quality and completeness.
        
        Args:
            project: Validated project data
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Check for missing optional but important data
        if not project.metadata.building_use:
            issues.append(ValidationIssue(
                level=ValidationLevel.INFO,
                field_path="metadata.building_use",
                message="Building use type not specified",
                suggestion="Specify building use for better compliance checking"
            ))
        
        # Check for rooms without area calculation
        rooms_without_area = [room for room in project.rooms if not room.area]
        if rooms_without_area:
            issues.append(ValidationIssue(
                level=ValidationLevel.INFO,
                field_path="rooms",
                message=f"{len(rooms_without_area)} rooms missing area calculation",
                suggestion="Calculate room areas for occupancy and fire load analysis"
            ))
        
        # Check for doors without fire rating
        doors_without_rating = [door for door in project.doors if door.is_egress and not door.fire_rating]
        if doors_without_rating:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                field_path="doors",
                message=f"{len(doors_without_rating)} emergency doors missing fire rating",
                suggestion="Specify fire resistance ratings for emergency doors"
            ))
        
        # Check fire equipment inspection dates
        outdated_equipment = []
        current_date = datetime.now()
        for equipment in project.fire_equipment:
            if equipment.last_inspection:
                days_since_inspection = (current_date - equipment.last_inspection).days
                if days_since_inspection > 365:  # More than 1 year
                    outdated_equipment.append(equipment)
        
        if outdated_equipment:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                field_path="fire_equipment",
                message=f"{len(outdated_equipment)} fire equipment items need inspection",
                suggestion="Update inspection dates for fire safety equipment"
            ))
        
        return issues
    
    def _has_nearby_exit(self, room: Room, project: Project) -> bool:
        """
        Check if a room has a nearby emergency exit.
        
        Args:
            room: Room to check
            project: Project data
            
        Returns:
            True if room has nearby exit access
        """
        # Simplified check - in practice would calculate actual distances
        egress_doors = [door for door in project.doors if door.is_egress]
        
        if not egress_doors:
            return False
        
        # Check if room has any doors (connected to circulation)
        connected_doors = [door for door in project.doors if room.id in (door.connected_rooms or [])]
        
        return len(connected_doors) > 0
    
    def _get_error_suggestion(self, error: Dict[str, Any]) -> Optional[str]:
        """
        Get helpful suggestions for common validation errors.
        
        Args:
            error: Pydantic validation error
            
        Returns:
            Helpful suggestion string or None
        """
        error_type = error.get('type', '')
        field_name = error.get('loc', [])[-1] if error.get('loc') else ''
        
        suggestions = {
            'value_error.missing': f"Add required field '{field_name}'",
            'type_error.none.not_allowed': f"Field '{field_name}' cannot be null",
            'value_error.any_str.min_length': f"Field '{field_name}' must not be empty",
            'value_error.number.not_gt': f"Field '{field_name}' must be greater than the minimum value",
            'value_error.list.min_items': f"List '{field_name}' must contain at least the minimum number of items",
        }
        
        return suggestions.get(error_type)
    
    def validate_multiple_files(self, file_paths: List[Path]) -> List[ValidationResult]:
        """
        Validate multiple JSON files.
        
        Args:
            file_paths: List of file paths to validate
            
        Returns:
            List of ValidationResult objects
        """
        results = []
        
        for file_path in file_paths:
            self.logger.info(f"Validating: {file_path}")
            result = self.validate_file(file_path)
            results.append(result)
        
        return results
    
    def print_validation_report(self, results: List[ValidationResult], 
                               show_warnings: bool = True, show_info: bool = False) -> None:
        """
        Print a formatted validation report.
        
        Args:
            results: List of validation results
            show_warnings: Whether to show warning-level issues
            show_info: Whether to show info-level issues
        """
        total_files = len(results)
        valid_files = sum(1 for r in results if r.is_valid)
        total_errors = sum(len([i for i in r.issues if i.level == ValidationLevel.ERROR]) for r in results)
        total_warnings = sum(len([i for i in r.issues if i.level == ValidationLevel.WARNING]) for r in results)
        
        print("\n" + "="*80)
        print("VALIDATION REPORT")
        print("="*80)
        print(f"Files processed: {total_files}")
        print(f"Valid files: {valid_files}")
        print(f"Invalid files: {total_files - valid_files}")
        print(f"Total errors: {total_errors}")
        print(f"Total warnings: {total_warnings}")
        print()
        
        for result in results:
            print(f"File: {result.file_path}")
            print(f"Status: {'✓ VALID' if result.is_valid else '✗ INVALID'}")
            print(f"Validation time: {result.validation_time:.3f}s")
            
            if result.issues:
                print("Issues:")
                for issue in result.issues:
                    if issue.level == ValidationLevel.ERROR:
                        print(f"  ❌ ERROR: {issue.field_path}: {issue.message}")
                    elif issue.level == ValidationLevel.WARNING and show_warnings:
                        print(f"  ⚠️  WARNING: {issue.field_path}: {issue.message}")
                    elif issue.level == ValidationLevel.INFO and show_info:
                        print(f"  ℹ️  INFO: {issue.field_path}: {issue.message}")
                    
                    if issue.suggestion:
                        print(f"      💡 Suggestion: {issue.suggestion}")
            
            print("-" * 40)
        
        if valid_files == total_files:
            print("🎉 All files validated successfully!")
        else:
            print(f"⚠️ {total_files - valid_files} file(s) failed validation")


def find_json_files(directory: Path, pattern: str = "*.json") -> List[Path]:
    """
    Find JSON files in a directory.
    
    Args:
        directory: Directory to search
        pattern: File pattern to match
        
    Returns:
        List of found JSON file paths
    """
    if not directory.exists():
        return []
    
    return list(directory.glob(pattern))


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Validate JSON files against AEC building data schemas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate single file
  python src/extraction/json_validator.py --file data/extracted/building.json
  
  # Validate all JSON files in directory
  python src/extraction/json_validator.py --directory data/extracted/
  
  # Strict validation (warnings as errors)
  python src/extraction/json_validator.py --file building.json --strict
  
  # Show all issue levels
  python src/extraction/json_validator.py --directory data/ --warnings --info
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--file', '-f',
        type=Path,
        help='Single JSON file to validate'
    )
    input_group.add_argument(
        '--directory', '-d',
        type=Path,
        help='Directory containing JSON files to validate'
    )
    
    # Validation options
    parser.add_argument(
        '--strict', '-s',
        action='store_true',
        help='Strict mode: treat warnings as errors'
    )
    
    parser.add_argument(
        '--pattern', '-p',
        type=str,
        default='*.json',
        help='File pattern for directory search (default: *.json)'
    )
    
    # Output options
    parser.add_argument(
        '--warnings', '-w',
        action='store_true',
        help='Show warnings in output'
    )
    
    parser.add_argument(
        '--info', '-i',
        action='store_true',
        help='Show info messages in output'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    # Create validator
    validator = JSONValidator(strict_mode=args.strict)
    
    # Determine files to validate
    files_to_validate = []
    
    if args.file:
        files_to_validate = [args.file]
    else:
        files_to_validate = find_json_files(args.directory, args.pattern)
        if not files_to_validate:
            print(f"No JSON files found in {args.directory} matching pattern '{args.pattern}'")
            sys.exit(1)
    
    # Validate files
    print(f"Validating {len(files_to_validate)} file(s)...")
    results = validator.validate_multiple_files(files_to_validate)
    
    # Print report
    validator.print_validation_report(
        results, 
        show_warnings=args.warnings,
        show_info=args.info
    )
    
    # Exit with appropriate code
    failed_count = sum(1 for r in results if not r.is_valid)
    if failed_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()