#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# USER SETTINGS (edit to match your TLC upload script)
# ============================================================

# LOGGING 
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
RUN_ID="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/bq_load_${RUN_ID}.log"

# GCP / BigQuery
# ============================================================
# Replace following  all variables with yours
  # PROJECT_ID
  # LOCATION 
  # BQ_DATASET
  # BQ_TABLE_PREFIX 
  # BUCKET
  # DEST_PREFIX
  # MONTHS
# ============================================================

PROJECT_ID="${PROJECT_ID:-bus673}"
# Dataset location
LOCATION="${LOCATION:-US}"                 

BQ_DATASET="${BQ_DATASET:-bus673_tlc}"
# tables: yellow_2023_01, yellow_2023_02, ...
BQ_TABLE_PREFIX="${BQ_TABLE_PREFIX:-yellow_2023_}"  

# GCS staging (must match where fetch_and_upload_tlc.sh uploaded files)
BUCKET="${BUCKET:-gs://bus673_e2_data}"
DEST_PREFIX="${DEST_PREFIX:-tlc/yellow/2023}"

# Months to load (match your fetch/upload months)
MONTHS=(01 02 03 04 05 06 07 08 09 10 11 12)

echo "============================================================"
echo "BigQuery load (from existing GCS Parquet)"
echo "Project : ${PROJECT_ID}"
echo "Location: ${LOCATION}"
echo "Dataset : ${BQ_DATASET}"
echo "Bucket  : ${BUCKET}"
echo "Prefix  : ${DEST_PREFIX}"
echo "Months  : ${MONTHS[*]}"
echo "============================================================"

# ============================================================
# Create dataset in BQ
# Display output in terminal AND save to a log file 
# ============================================================
bq --location="${LOCATION}" mk -d "${PROJECT_ID}:${BQ_DATASET}" 2>&1 | tee -a "$LOG_FILE" || true

# ============================================================
# Load each month from GCS to BigQuery (Parquet files)
# ============================================================
for M in "${MONTHS[@]}"; do
  # Adjust naming if your uploaded names differ
  OBJECT="${BUCKET}/${DEST_PREFIX}/yellow_tripdata_2023-${M}.parquet"
  TABLE="${PROJECT_ID}:${BQ_DATASET}.${BQ_TABLE_PREFIX}${M}"

  echo "----------------------------------------"
  echo "Month: ${M}"
  echo "GCS  : ${OBJECT}"
  echo "BQ   : ${TABLE}"

  bq load \
    --location="${LOCATION}" \
    --replace \
    --source_format=PARQUET \
    "${TABLE}" \
    "${OBJECT}"

  # Quick check
  bq head -n 3 "${TABLE}" || true
done

echo "Done loading all months into BigQuery."
