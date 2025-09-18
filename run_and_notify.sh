#!/bin/bash

# Galformer automated run with email notification
# Usage: ./run_and_notify.sh [your_email@domain.com]

EMAIL=${1:-"archerdw@example.com"}  # Replace with your actual email
PROJECT_DIR="/u/archerdw/galformer_project"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$PROJECT_DIR/cron_logs/galformer_run_$TIMESTAMP.log"

# Create log directory if it doesn't exist
mkdir -p "$PROJECT_DIR/cron_logs"

cd "$PROJECT_DIR"

echo "Starting Galformer run at $(date)" > "$LOG_FILE"

# Submit the SLURM job and capture job ID
JOB_ID=$(sbatch run_galformer.slurm | awk '{print $4}')
echo "Submitted SLURM job ID: $JOB_ID" >> "$LOG_FILE"

# Wait for job to complete
echo "Waiting for job $JOB_ID to complete..." >> "$LOG_FILE"
while squeue -j "$JOB_ID" &>/dev/null; do
    sleep 30
done

# Check job status
JOB_STATUS=$(sacct -j "$JOB_ID" --format=State --noheader --parsable2 | head -1)
echo "Job $JOB_ID completed with status: $JOB_STATUS" >> "$LOG_FILE"

# Find the most recent output file
OUTPUT_FILE=$(ls -t galformer_*.out 2>/dev/null | head -1)

# Prepare email content
if [[ "$JOB_STATUS" == "COMPLETED" ]]; then
    SUBJECT="✅ Galformer Job $JOB_ID Completed Successfully"
    
    # Extract key results from output file if it exists
    if [[ -f "$OUTPUT_FILE" ]]; then
        echo -e "\n--- Job Output Summary ---" >> "$LOG_FILE"
        tail -20 "$OUTPUT_FILE" >> "$LOG_FILE"
        
        # Look for accuracy metrics
        if grep -q "accuracy metrics" "$OUTPUT_FILE"; then
            echo -e "\n--- Accuracy Metrics ---" >> "$LOG_FILE"
            grep -A 5 "accuracy metrics" "$OUTPUT_FILE" >> "$LOG_FILE"
        fi
    fi
else
    SUBJECT="❌ Galformer Job $JOB_ID Failed"
    echo "Job failed. Check SLURM logs for details." >> "$LOG_FILE"
fi

# Send email notification
mail -s "$SUBJECT" "$EMAIL" < "$LOG_FILE"

echo "Email notification sent to $EMAIL" >> "$LOG_FILE"
echo "Run completed at $(date)" >> "$LOG_FILE"
