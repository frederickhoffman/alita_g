from typing import Any, Dict, Optional

import wandb


class WandBMonitor:
    def __init__(self, project: str = "alita-g", entity: Optional[str] = None):
        self.project = project
        self.entity = entity

    def start_run(self, name: str, config: Dict[str, Any]) -> None:
        wandb.init(project=self.project, entity=self.entity, name=name, config=config)

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        wandb.log(metrics)

    def log_prompt(
        self, prompt: str, completion: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Logs a prompt and completion as a wandb Table."""
        # Simple logging for now, can be expanded to Tables
        wandb.log({"prompt": prompt, "completion": completion, "metadata": metadata or {}})

    def finish(self) -> None:
        wandb.finish()
