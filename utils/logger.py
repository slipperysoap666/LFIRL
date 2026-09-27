import os
import csv
from datetime import datetime
import torch


def _sanitize_env_id(env_id: str) -> str:
    """Replace filesystem-hostile chars for folder names."""
    return env_id.replace("/", "_").replace(":", "_")


class Logger:
    """
    Lightweight experiment logger.

    Behavior:
      1) Under results_root, create a folder named by env_id (sanitized).
      2) Inside it, create a subfolder named by the current timestamp (to seconds).
      3) Initialize results.csv with header: ["env steps","q_loss","v_loss","r_loss"].
      4) Provide log(step, q, v, r) to append one row per training step (100 total).
      5) finalize(total_time_sec) appends a last line with total running time.
      6) save_models(...) stores q/v/r state_dicts into the run folder.

    Notes about "special rows":
      - For scalar metrics that do not fit the main training schema (e.g., SAC evaluation),
        we write a special row whose first cell is a string label, and the scalar is placed
        into the 'q_loss' column. This keeps backward compatibility with the original CSV
        header while allowing extra metrics to be recorded before finalize().
    """

    def __init__(self, env_id: str, results_root: str):
        self.env_id = env_id
        self.results_root = results_root

        env_dir = os.path.join(self.results_root, _sanitize_env_id(self.env_id))
        os.makedirs(env_dir, exist_ok=True)

        # Timestamp precise to second to avoid collisions.
        self.run_start = datetime.now()
        self.run_dir = os.path.join(env_dir, self.run_start.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.run_dir, exist_ok=True)

        self.csv_path = os.path.join(self.run_dir, "results.csv")
        # Initialize CSV with header row
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["env steps", "q_loss", "v_loss", "r_loss"])

    def log(self, step: int, q_loss: float, v_loss: float, r_loss: float) -> None:
        """Append one result row corresponding to a single training iteration."""
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([int(step), float(q_loss), float(v_loss), float(r_loss)])

    def log_scalar(self, name: str, value: float) -> None:
        """
        Append a 'special row' with an arbitrary scalar metric.
        The metric name goes into the first column, and the numeric value goes into the 'q_loss' column.
        This is intentionally consistent with finalize() and keeps the CSV header unchanged.
        """
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([str(name), f"{float(value):.6f}", "", ""])

    def log_reward(self, value: float, label: str = "sac_succ_rate") -> None:
        """
        Convenience wrapper to record the SAC evaluation mean return into results.csv
        as a special row. Call this BEFORE finalize().
        """
        self.log_scalar(label, value)

    def finalize(self, total_time_sec: float) -> None:
        """Append a final row that records total wall-clock time (seconds)."""
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            # Put total time into the 'q_loss' column by convention.
            writer.writerow(["total_time_sec", f"{float(total_time_sec):.4f}", "", ""])

    def save_models(self, *, q_net, v_net, r_net) -> None:
        """Persist trained model weights into the run folder."""
        q_path = os.path.join(self.run_dir, "q_net.pt")
        v_path = os.path.join(self.run_dir, "v_net.pt")
        r_path = os.path.join(self.run_dir, "r_net.pt")
        torch.save(q_net.state_dict(), q_path)
        torch.save(v_net.state_dict(), v_path)
        torch.save(r_net.state_dict(), r_path)
