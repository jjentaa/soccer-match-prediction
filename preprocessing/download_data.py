"""Step 1 - download the raw datasets into preprocessing/raw/ (not committed).

1. European Soccer Database (Kaggle: hugomathien/soccer) -> raw/european_soccer/database.sqlite
2. football-data.co.uk per-league/season match stats     -> raw/football_data/{season}_{code}.csv

Usage: python preprocessing/download_data.py [--only esd|fd]
"""

import argparse
import time
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"
ESD_SQLITE = RAW / "european_soccer" / "database.sqlite"
FOOTBALL_DATA_DIR = RAW / "football_data"

ESD_URL = "https://www.kaggle.com/api/v1/datasets/download/hugomathien/soccer"
FD_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"

# Seasons covered by the European Soccer Database (2008/09 - 2015/16)
SEASONS = ["0809", "0910", "1011", "1112", "1213", "1314", "1415", "1516"]
# football-data.co.uk division codes for the ESD leagues.
# Poland and Switzerland are not published in this per-season format.
LEAGUE_CODES = {
    "England Premier League": "E0",
    "France Ligue 1": "F1",
    "Germany 1. Bundesliga": "D1",
    "Italy Serie A": "I1",
    "Spain LIGA BBVA": "SP1",
    "Netherlands Eredivisie": "N1",
    "Belgium Jupiler League": "B1",
    "Portugal Liga ZON Sagres": "P1",
    "Scotland Premier League": "SC0",
}


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def download_esd() -> None:
    if ESD_SQLITE.exists():
        print(f"skip  {ESD_SQLITE} (exists)")
        return
    archive = ESD_SQLITE.parent / "archive.zip"
    print(f"fetch {ESD_URL}")
    fetch(ESD_URL, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(ESD_SQLITE.parent)
    archive.unlink()
    print(f"ok    {ESD_SQLITE} ({ESD_SQLITE.stat().st_size / 1e6:.0f} MB)")


def download_football_data() -> None:
    for season in SEASONS:
        for code in LEAGUE_CODES.values():
            dest = FOOTBALL_DATA_DIR / f"{season}_{code}.csv"
            if dest.exists():
                continue
            fetch(FD_URL.format(season=season, code=code), dest)
            print(f"ok    {dest.name}")
            time.sleep(0.3)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["esd", "fd"])
    args = p.parse_args()
    if args.only in (None, "esd"):
        download_esd()
    if args.only in (None, "fd"):
        download_football_data()
