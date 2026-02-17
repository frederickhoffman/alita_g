from typing import Any, Dict

import wandb


class WandBMonitor:
    def __init__(self, project: str = "alita-g", entity: str = None):
        self.project = project
        self.entity = entity

    def start_run(self, name: str, config: Dict[str, Any]):
        wandb.init(project=self.project, entity=self.entity, name=name, config=config)

    def log_metrics(self, metrics: Dict[str, Any]):
        wandb.log(metrics)

    def log_prompt(self, prompt: str, completion: str, metadata: Dict[str, Any] = None):
        """Logs a prompt and completion as a wandb Table."""
        # Simple logging for now, can be expanded to Tables
        wandb.log({
            "prompt": prompt,
            "completion": completion,
            "metadata": metadata or {}
        })

    def finish(self):
        wandb.finish()
