import yaml
import subprocess
import json
import re
from pathlib import Path

CONFIG_PATH = "config.yaml"
RESULTS_PATH = "ablation_results.json"

configs = [
    {"name": "Stage 1-5 only", "enable_id_repair": False, "resurrection_enabled": False},
    {"name": "Stage 1-5 + IDRepair", "enable_id_repair": True, "resurrection_enabled": False},
    {"name": "Stage 1-5 + Resurrection", "enable_id_repair": False, "resurrection_enabled": True},
    {"name": "Stage 1-5 + IDRepair + Resurrection", "enable_id_repair": True, "resurrection_enabled": True},
]

results = {}

for c in configs:
    print(f"\n--- Running Configuration: {c['name']} ---")
    
    # Update config.yaml
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        
    if "recovery" not in config:
        config["recovery"] = {}
    config["recovery"]["enable_id_repair"] = c["enable_id_repair"]
    
    if "resurrection" not in config:
        config["resurrection"] = {}
    config["resurrection"]["enabled"] = c["resurrection_enabled"]
    
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f)
        
    # Run main.py
    print("Running tracker...")
    subprocess.run(["python", "src/main.py", "--input", "data/video/short2.mp4", "--output-csv", "outputs/csv/short2_ablation.csv"], check=True)
    
    # Run merge_audit.py
    print("Running merge audit...")
    audit_res = subprocess.run(["python", "src/merge_audit.py", "--csv", "outputs/csv/short2_ablation.csv", "--video", "data/video/short2.mp4"], capture_output=True, text=True)
    
    # Extract metrics from outputs/metrics/diagnostics_summary.json
    with open("outputs/metrics/diagnostics_summary.json", "r") as f:
        diag = json.load(f)
        
    # Parse merge audit output
    audit_output = audit_res.stderr + "\n" + audit_res.stdout
    total_tracks_match = re.search(r"Total Tracks:\s*(\d+)", audit_output)
    suspicious_match = re.search(r"Suspicious Tracks:\s*(\d+)", audit_output)
    
    # Parse anomalies (count occurrences in Top 10 list or wherever)
    app_anomalies = len(re.findall(r"appearance_jump", audit_output))
    vel_anomalies = len(re.findall(r"velocity_jump", audit_output))
    
    res = {
        "Unique Tracks": diag.get("tracking_diagnostics", {}).get("total_unique_tracks", 0),
        "Fragmentation Gaps": diag.get("tracking_diagnostics", {}).get("track_fragmentation_gaps", 0),
        "Suspicious Tracks": int(suspicious_match.group(1)) if suspicious_match else 0,
        "Appearance Anomalies": app_anomalies,
        "Teleportation Anomalies": vel_anomalies
    }
    
    # For unique tracks if diag structure is different
    if res["Unique Tracks"] == 0:
        res["Unique Tracks"] = diag.get("processing_summary", {}).get("unique_track_ids", 0)
        
    # For frag gaps if diag structure is different
    if res["Fragmentation Gaps"] == 0:
        res["Fragmentation Gaps"] = diag.get("tracking_diagnostics", {}).get("fragmentation", {}).get("overall_gaps", 0)
        if res["Fragmentation Gaps"] == 0:
            res["Fragmentation Gaps"] = diag.get("tracking_diagnostics", {}).get("overall_fragmentation_gaps", 0)
            
    results[c["name"]] = res
    print(res)

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print("\nDone. Results saved to", RESULTS_PATH)
