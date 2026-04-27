# DeepVision Training Worker

Python worker used by the Spring backend for real training in mode B.

Spring starts this script as a subprocess, passes a JSON request file, and reads
newline-delimited JSON metrics from stdout.

## Run Manually

```powershell
cd backend/python-training
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The script is normally launched by Spring:

```powershell
python -B training_worker.py --request path\to\request.json
```
