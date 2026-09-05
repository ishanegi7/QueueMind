# QueueMind Data Directory Guide

This directory holds the raw and processed datasets used by the QueueMind patient flow intelligence system.

## Strict Data Governance & Compliance

> [!CAUTION]
> **Zero Clinical Data in Git:** Under the MIMIC-IV-ED Data Use Agreement (DUA) and PhysioNet ethical guidelines, real clinical records, de-identified patient data, derived features, and intermediate tables must **never** be committed to version control or distributed publicly.

- `data/raw/*` is strictly ignored by `.gitignore` (except `.gitkeep` and this guide).
- `data/processed/*` is strictly ignored by `.gitignore` (except `.gitkeep`).
- Never upload, distribute, or share downloaded PhysioNet files.
- Never commit synthetic clinical data masquerading as real healthcare data.

---

## Accessing the MIMIC-IV-ED Dataset

QueueMind is built to work with the official, de-identified [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/) dataset hosted on PhysioNet.

To obtain authorized access:
1. **Register on PhysioNet:** Create an account at [physionet.org](https://physionet.org/).
2. **Complete Human Subjects Training:** Successfully pass the CITI "Data or Specimens Only Research" course.
3. **Sign the Data Use Agreement (DUA):** Adhere to the strict terms prohibiting re-identification or sharing of data.
4. **Request Access:** Submit an access request for the MIMIC-IV-ED project.

---

## Directory Organization

When access is approved, place the downloaded MIMIC-IV-ED files into `data/raw/`:

```text
data/
├── README.md              # This guide (tracked in Git)
├── raw/                   # Raw MIMIC-IV-ED tables (IGNORED in Git)
│   ├── .gitkeep
│   ├── edstays.csv.gz     (or edstays.csv / edstays.parquet)
│   ├── diagnosis.csv.gz
│   ├── medrecon.csv.gz
│   ├── pyxis.csv.gz
│   ├── triage.csv.gz
│   └── vitalsign.csv.gz
└── processed/             # Cleaned, standardized datasets (IGNORED in Git)
    └── .gitkeep
```

### Supported Formats
`src.queuemind.data.loader.MIMICDataLoader` automatically recognizes and prioritizes:
1. Apache Parquet (`.parquet`)
2. Gzip-compressed CSV (`.csv.gz`)
3. Standard CSV (`.csv`)

### Environment Variable Configuration
If you store data outside this repository (e.g. on external SSD, shared secure NAS, or cloud storage mount), set the path in your `.env` file:
```bash
MIMIC_DATA_DIR="/path/to/your/secure/mimic-iv-ed"
PROCESSED_DATA_DIR="/path/to/your/secure/processed"
```
