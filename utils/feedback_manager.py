import os
import json
import csv
import time
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

class FeedbackManager:
    """
    Manages the persistence of user feedback and system evaluations.
    Saves results in both JSONL (for programatic access) and CSV (for human review) formats.
    """

    def __init__(self, feedback_file: str = "feedback.jsonl"):
        self.feedback_file = Path(feedback_file)
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)

    def save_feedback(
        self, 
        query: str, 
        configuration: Dict[str, Any], 
        retrieved_artifacts: List[str], 
        generated_response: str, 
        feedback: Any, 
        feedback_text: str = ""
    ):
        """
        Creates a new feedback entry with unique ID and timestamp.
        Appends the record to both the JSON-Lines and CSV log files.
        """
        
        # Unique identifier based on milliseconds
        interaction_id = f"int_{int(time.time() * 1000)}"
        timestamp = datetime.now().isoformat()

        entry = {
            "interaction_id": interaction_id,
            "timestamp": timestamp,
            "query": query,
            "configuration": configuration,
            "retrieved_artifacts": retrieved_artifacts,
            "generated_response": generated_response,
            "feedback": feedback,
            "feedback_text": feedback_text
        }

        # 1. Store the raw record in JSONL format
        with open(self.feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        
        # 2. Store the flattened record in CSV format
        csv_path = self.feedback_file.with_suffix(".csv")
        file_existed = csv_path.exists()
        
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=entry.keys())
            if not file_existed:
                writer.writeheader()
            
            # Complex objects are JSON-encoded for tabular storage
            csv_record = entry.copy()
            csv_record["configuration"] = json.dumps(csv_record["configuration"])
            csv_record["retrieved_artifacts"] = json.dumps(csv_record["retrieved_artifacts"])
            writer.writerow(csv_record)
            
        print(f"Feedback entry {interaction_id} has been recorded locally.")
        return interaction_id
