# FPL Data Scripts

A professional Python project for pulling and managing Fantasy Premier League (FPL) data with proper testing, linting, and CI/CD.

[![Python CI](https://github.com/wiesnerbernard/fpl/actions/workflows/ci.yml/badge.svg)](https://github.com/wiesnerbernard/fpl/actions/workflows/ci.yml)

## Project Structure

```
fpl/
├── src/                      # Source code
│   ├── __init__.py
│   ├── fpl_api.py           # FPL API client and data fetching
│   ├── data_processing.py   # Player data processing utilities
│   └── pull_fpl_to_sheet.py # Main script for data export
├── tests/                    # Unit tests
│   ├── __init__.py
│   ├── test_fpl_api.py
│   └── test_data_processing.py
├── .github/workflows/        # CI/CD workflows
│   └── ci.yml
├── instructions.md           # Python coding standards
├── pyproject.toml           # Project configuration
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
├── .flake8                  # Linting configuration
└── .gitignore               # Git ignore rules
```

## Installation

### Production

```bash
pip install -r requirements.txt
```

### Development

```bash
pip install -r requirements-dev.txt
```

## Usage

### Pull FPL Data to Spreadsheet

Run the main script to fetch current FPL data and export to Excel/CSV:

```bash
cd src
python pull_fpl_to_sheet.py
```

This will create:
- `fpl_players.xlsx` - Excel format with all player data
- `fpl_players.csv` - CSV backup

### Legacy Scripts

Complex scripts have been moved to the `legacy/` directory for reference:
- `legacy/pull_and_sync_scrypt.py` - Advanced Notion sync with player scoring (862 lines)

These can be refactored into the modular `src/` structure in future iterations.

## Development

### Run Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=src --cov-report=html
```

### Linting and Formatting

```bash
# Check code style
flake8 src tests

# Format code
black src tests

# Sort imports
isort src tests
```

### Code Standards

See [instructions.md](instructions.md) for detailed Python coding standards and best practices.

## Data Sources

- Fantasy Premier League API: https://fantasy.premierleague.com/api/

## Output Files

- `fpl_players.xlsx` - Excel format player data
- `fpl_players.csv` - CSV format player data
- `fpl_players_scored.csv` - Scored player data (from sync script)
- `fpl_players_debug.csv` - Debug output (from sync script)

## CI/CD

The project uses GitHub Actions for continuous integration:
- Runs tests on Python 3.9, 3.10, and 3.11
- Checks code formatting (black)
- Validates import order (isort)
- Lints with flake8
- Generates coverage reports

## License

This project is for personal use.
