import psutil
import time
import threading
from typing import Optional

class MemoryMonitor:
    """
    Context manager to monitor the peak RAM usage of a specific process.
    By default, it looks for the 'ollama' server process.
    """
    def __init__(self, process_name: str = "ollama"):
        self.process_name = process_name
        self.peak_memory_mb: float = 0.0
        self.keep_running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.target_pid: Optional[int] = self._find_pid()

    def _find_pid(self) -> Optional[int]:
        """Find the PID of the target process by name."""
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'] and self.process_name.lower() in proc.info['name'].lower():
                return proc.info['pid']
        return None

    def _monitor(self):
        """Polls the memory usage in a background thread."""
        if not self.target_pid:
            return
        
        try:
            process = psutil.Process(self.target_pid)
            while self.keep_running:
                mem_info = process.memory_info()
                rss_mb = mem_info.rss / (1024 * 1024)
                if rss_mb > self.peak_memory_mb:
                    self.peak_memory_mb = rss_mb
                time.sleep(0.1)  # Poll every 100ms
        except psutil.NoSuchProcess:
            pass

    def __enter__(self):
        self.keep_running = True
        self.peak_memory_mb = 0.0
        if self.target_pid:
            self.thread = threading.Thread(target=self._monitor)
            self.thread.start()
        else:
            print(f"Warning: Process '{self.process_name}' not found. Memory monitoring disabled.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.keep_running = False
        if self.thread:
            self.thread.join()

def calculate_throughput(eval_count: int, eval_duration_ns: int) -> float:
    """
    Calculate throughput in tokens per second.
    
    Args:
        eval_count: Number of generated tokens.
        eval_duration_ns: Time taken to generate tokens in nanoseconds.
        
    Returns:
        float: Tokens per second.
    """
    if not eval_duration_ns or eval_duration_ns == 0:
        return 0.0
    return eval_count / (eval_duration_ns / 1e9)

def calculate_latency_ms(eval_duration_ns: int) -> float:
    """
    Calculate latency in milliseconds.
    
    Args:
        eval_duration_ns: Duration in nanoseconds.
        
    Returns:
        float: Latency in milliseconds.
    """
    if not eval_duration_ns:
        return 0.0
    return eval_duration_ns / 1_000_000
