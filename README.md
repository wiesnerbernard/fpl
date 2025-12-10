# FPL Data Scripts

Python scripts for pulling and managing Fantasy Premier League (FPL) data.

## Scripts

- **pull_fpl_to_sheet.py** - Pull current FPL data and write to spreadsheet files
- **pull_script.py** - General FPL data pulling script
- **pull_and_sync_scrypt.py** - Pull and sync FPL data

## Requirements

```bash
pip install requests pandas python-dateutil openpyxl
```

## Usage

Run the main script to fetch FPL data:

```bash
python pull_fpl_to_sheet.py
```

## Data Sources

- Fantasy Premier League API: https://fantasy.premierleague.com/api/

## Output Files

- `fpl_players.xlsx` - Excel format player data
- `fpl_players.csv` - CSV format player data
- `fpl_players_scored.csv` - Scored player data
- `fpl_players_debug.csv` - Debug output

## License

This project is for personal use.
