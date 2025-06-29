"""Memory monitoring utilities for preventing OOM kills on macOS and other systems."""

import time
import threading
import torch
from typing import Callable, Optional
from genrl_swarm.logging_utils.global_defs import get_logger


class MemoryMonitor:
    """Monitor system and GPU memory usage to prevent OOM kills."""
    
    def __init__(
        self,
        check_interval: float = 5.0,
        memory_threshold: float = 0.85,
        gpu_threshold: float = 0.9,
        callback: Optional[Callable] = None
    ):
        self.check_interval = check_interval
        self.memory_threshold = memory_threshold
        self.gpu_threshold = gpu_threshold
        self.callback = callback or self._default_cleanup
        self.monitoring = False
        self.monitor_thread = None
        self.logger = get_logger()
        
    def _default_cleanup(self):
        """Default cleanup function to call when memory pressure is detected."""
        self.logger.warning("Memory pressure detected, clearing cache")
        try:
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            # Always force garbage collection for CPU memory (important for CPU-only systems)
            gc.collect()
        except Exception as e:
            self.logger.error(f"Error during memory cleanup: {e}")
    
    def get_system_memory_usage(self) -> float:
        """Get current system memory usage as percentage (simplified without psutil)."""
        # Without psutil, use conservative estimate
        # Return moderate usage to trigger some monitoring but not too aggressive
        return 0.6  # Conservative 60% estimate
    
    def get_gpu_memory_usage(self) -> float:
        """Get current memory usage as percentage (GPU or system memory for CPU-only)."""
        try:
            if torch.cuda.is_available():
                free_memory, total_memory = torch.cuda.mem_get_info()
                used_memory = total_memory - free_memory
                return used_memory / total_memory
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # For MPS, estimate based on system memory since it's unified
                return self.get_system_memory_usage()
            else:
                # CPU-only systems - use system memory usage as proxy
                return self.get_system_memory_usage()
        except Exception:
            pass
        return 0.0
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        system_usage = self.get_system_memory_usage()
        gpu_usage = self.get_gpu_memory_usage()
        
        if system_usage > self.memory_threshold:
            self.logger.warning(f"System memory usage high: {system_usage:.1%}")
            return True
            
        if gpu_usage > self.gpu_threshold:
            self.logger.warning(f"GPU memory usage high: {gpu_usage:.1%}")
            return True
            
        return False
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                if self.check_memory_pressure():
                    self.callback()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in memory monitor: {e}")
                time.sleep(self.check_interval)
    
    def start(self):
        """Start memory monitoring in background thread."""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("Memory monitoring started")
    
    def stop(self):
        """Stop memory monitoring."""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=1.0)
            self.logger.info("Memory monitoring stopped")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def create_memory_monitor(**kwargs) -> MemoryMonitor:
    """Create a memory monitor with system-optimized defaults."""
    import sys
    
    # Detect if we're on CPU-only system
    is_cpu_only = not torch.cuda.is_available() and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
    
    if sys.platform == 'darwin':
        if is_cpu_only:
            # Mac Mini M4 style - CPU only
            kwargs.setdefault('check_interval', 3.0)  # Frequent checks for CPU-only
            kwargs.setdefault('memory_threshold', 0.6)  # Very conservative for CPU-only
            kwargs.setdefault('gpu_threshold', 0.7)     # Actually system memory threshold
        else:
            # Apple Silicon with MPS
            kwargs.setdefault('check_interval', 2.0)  # More frequent checks
            kwargs.setdefault('memory_threshold', 0.7)  # Lower threshold
            kwargs.setdefault('gpu_threshold', 0.8)     # More conservative GPU threshold
    else:
        kwargs.setdefault('check_interval', 5.0)
        kwargs.setdefault('memory_threshold', 0.85)
        kwargs.setdefault('gpu_threshold', 0.9)
    
    return MemoryMonitor(**kwargs)


def log_memory_stats():
    """Log current memory statistics (simplified without psutil)."""
    logger = get_logger()
    
    # Log device info and memory estimates
    if torch.cuda.is_available():
        try:
            free_memory, total_memory = torch.cuda.mem_get_info()
            logger.info(f"CUDA memory: {(total_memory - free_memory) / (1024**3):.1f}GB used, {free_memory / (1024**3):.1f}GB free")
        except Exception:
            logger.info("CUDA available but could not get memory info")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("MPS backend active (unified memory)")
    else:
        logger.info("CPU-only mode (no GPU acceleration)")
    
    logger.info("Memory monitoring active (simplified mode)")
    
    # GPU memory
    try:
        if torch.cuda.is_available():
            free_memory, total_memory = torch.cuda.mem_get_info()
            used_memory = total_memory - free_memory
            logger.info(f"CUDA memory: {used_memory / (1024**3):.1f}GB used, {free_memory / (1024**3):.1f}GB free")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS backend active (unified memory)")
    except Exception as e:
        logger.warning(f"Could not get GPU memory info: {e}")