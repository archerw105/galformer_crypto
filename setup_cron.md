# Galformer Automated Cron Job Setup

## Quick Setup Instructions

### 1. Make the script executable:
```bash
chmod +x /u/archerdw/galformer_project/run_and_notify.sh
```

### 2. Edit your crontab:
```bash
crontab -e
```

### 3. Add one of these cron entries:

**Daily at 2 AM:**
```
0 2 * * * /u/archerdw/galformer_project/run_and_notify.sh your_email@domain.com
```

**Weekly on Sundays at 1 AM:**
```
0 1 * * 0 /u/archerdw/galformer_project/run_and_notify.sh your_email@domain.com
```

**Every 6 hours:**
```
0 */6 * * * /u/archerdw/galformer_project/run_and_notify.sh your_email@domain.com
```

### 4. View current cron jobs:
```bash
crontab -l
```

### 5. Check logs:
Logs will be saved in `/u/archerdw/galformer_project/cron_logs/`

## What the script does:
- Submits your SLURM job
- Waits for completion
- Extracts key results (accuracy metrics)
- Sends email with success/failure status
- Saves detailed logs

## Email Requirements:
Make sure your system has `mail` command configured. If not available, you can replace the mail command in the script with:
```bash
# Alternative: use sendmail or other email tools
echo "$LOG_FILE" | sendmail "$EMAIL"
```
