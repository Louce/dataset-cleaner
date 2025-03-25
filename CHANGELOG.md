# Changelog

All notable changes to the Dataset Cleaner will be documented in this file.

## [1.1.0] - 2025-04-15

### Added
- Excel file support (.xlsx, .xls) for data import
- Sheet selection for Excel workbooks
- Excel-specific import settings:
  - Header row selection
  - Specific column selection
  - Formatted value conversion
- Enhanced Excel export with formatting options:
  - Custom sheet naming with validation
  - Header row freezing
  - Automatic filtering
  - Excel table formatting with styles
- Updated application title and UI elements to reflect multi-format support
- Enhanced file type detection and validation
- Added xlsxwriter dependency for enhanced Excel export capabilities

### Fixed
- Issue with duplicate UI elements in Excel sheet selection
- Improved exception handling for Excel operations
- Added validation for Excel sheet names and table names
- Ensured proper closing of Excel writer objects

## [1.0.0] - 2025-03-24

### Added
- Initial release of CSV Dataset Cleaner
- Dynamic dataset handling with automatic data type detection
- Interactive missing value editing with multiple strategies:
  - Synergy Strategy using KNN imputation for numeric columns and conditional mode for categorical columns
  - Statistical methods (mean, median, mode)
  - Custom value replacement with enhanced UI
- Advanced outlier detection and handling using IQR and Z-score methods
- Data consistency checks with automatic issue detection
- Interactive visualization dashboards
- Flexible export options (CSV, Excel, JSON, Pickle)
- Comprehensive documentation and testing

### Fixed
- Fixed indentation issues in the data consistency section
- Added tracking for outlier changes across all columns
- Improved visualization of changes in the Changes Visualization tab

## Future Planned Features
- Data profiling with automated insights
- Advanced data cleaning pipelines
- Scheduled cleaning jobs
- Integration with cloud storage providers
- API for programmatic access 