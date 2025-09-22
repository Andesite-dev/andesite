# Andesite

**Geostatistical Analysis and Mining Estimation Software**

Andesite is a comprehensive Python package for geostatistical analysis and resource estimation in mining applications. It provides a complete workflow from drill hole data processing to kriging-based estimation and validation.

## Overview

Andesite implements industry-standard geostatistical methods used in mining resource estimation, with particular focus on:

- **Drill hole data compositing** (length-weighted and categorical)
- **Experimental and theoretical variogram modeling**
- **Kriging-based resource estimation** (OK, UK, CK, IK)
- **Statistical validation and cross-validation**
- **Resource classification and uncertainty quantification**

## Key Features

- **Complete compositing pipeline** for numeric and categorical variables
- **GSLIB format compatibility** with industry-standard tools
- **Multiple kriging algorithms** including ordinary, universal, co-kriging, and indicator kriging
- **Interactive visualizations** using Plotly for analysis and reporting
- **Robust validation tools** including cross-validation and statistical analysis
- **Pass classification** based on estimation uncertainty
- **Integration with external binaries** (KT3D, GAMV) for performance

## Package Structure

### Core Modules

#### `andesite.composite`
**Drill hole data compositing operations**
- `composite.py` - Main compositing class with length-weighted averaging
- `numeric_compositing.py` - Numeric variable compositing algorithms
- `categoric_compositing.py` - Categorical variable compositing with dominant class logic
- `composite_exceptions.py` - Custom exception handling

#### `andesite.estimations`
**Resource estimation algorithms**
- `kriging.py` - Ordinary kriging implementation
- `ugkriging.py` - Universal kriging algorithms
- `cokriging.py` - Co-kriging for multiple variables
- `indicatorkriging.py` - Indicator kriging for threshold analysis
- `ugcokriging.py` - Universal co-kriging
- `estimation_exceptions.py` - Estimation error handling

#### `andesite.variography`
**Variogram modeling and analysis**
- `experimental.py` - Experimental variogram calculation
- `modeling.py` - Theoretical variogram model fitting
- `maps.py` - Variogram map generation and analysis

#### `andesite.analisis`
**Statistical analysis and validation tools**
- `cross_validation.py` - Cross-validation for estimation quality assessment
- `histogram.py` - Statistical distribution analysis
- `slicer_view_model.py` - Interactive data slicing and visualization

#### `andesite.clasification`
**Resource classification based on estimation uncertainty**
- `pass_classification.py` - Pass/fail classification algorithms
- `varkrig_classification.py` - Kriging variance-based classification
- `validation.py` - Classification validation tools

#### `andesite.datafiles`
**Data file handling and grid operations**
- `grid.py` - Grid data structures and operations for spatial data

#### `andesite.utils`
**Common utilities and helper functions**
- `files.py` - File I/O operations and format handling
- `log.py` - Logging configuration and utilities
- `manipulations.py` - Data manipulation and transformation functions
- `version.py` - Git-based version management
- `_globals.py` - Global constants and configuration

## Quick Start

### Basic Compositing Workflow

```python
import andesite.composite as comp

# Load drill hole data
TODO

# Configure compositing parameters
TODO

# Execute compositing
TODO
```

### Variogram Analysis

```python
import andesite.variography as vario

# Calculate experimental variogram
TODO

# Fit theoretical model
TODO
```

### Kriging Estimation

```python
import andesite.estimations as est

# Setup ordinary kriging
TODO

# Run estimation
TODO
```

## Advanced Features

### Cross-Validation
```python
from andesite.analisis import cross_validation

TODO
```

### Resource Classification
```python
from andesite.clasification import varkrig_classification

TODO
```

## Integration with GSLIB

Andesite includes pre-compiled GSLIB executables for enhanced performance:

- **KT3D** - 3D kriging estimation
- **GAMV** - Experimental variogram calculation

These binaries are automatically included in the package installation under `andesite/utils/bin/`.

## Testing

The package includes comprehensive unit tests:

```bash
pytest andesite/tests/
```

Test coverage includes:
- Numeric and categorical compositing validation
- Kriging algorithm accuracy
- Variogram modeling consistency
- Data integrity checks

## Documentation

For detailed API documentation and examples, see the individual module docstrings and the included test files which demonstrate usage patterns.

## License

See `LICENSE.txt` for license information.

## Contact

**ANDESITE SpA**
- Website: http://www.andesite.cl/
- Email: soporte@andesite.cl

---

*Andesite - Professional geostatistical software for mining resource estimation*