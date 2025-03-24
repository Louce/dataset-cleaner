# Changelog

All notable changes to the CSV Dataset Cleaner will be documented in this file.

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