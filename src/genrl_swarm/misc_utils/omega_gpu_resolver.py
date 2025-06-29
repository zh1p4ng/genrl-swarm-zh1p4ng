import random
import torch
from omegaconf import OmegaConf

def get_gpu_vram():
    """Returns the total memory available for models in GiB (GPU VRAM or system RAM for CPU-only)."""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        return total_memory / (1024**3) # Convert bytes to GiB
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon unified memory - use conservative estimate
        # Use system info from resource module (part of stdlib)
        try:
            import resource
            # Get system memory via resource limit (not perfect but available)
            max_memory = resource.getrlimit(resource.RLIMIT_AS)[0]
            if max_memory != resource.RLIM_INFINITY:
                # Reserve 25% for system, use 75% for model
                available_memory = max_memory * 0.75
                return available_memory / (1024**3)
        except (ImportError, OSError):
            pass
        # Fallback: conservative estimate for Apple Silicon Macs
        return 16.0  # Conservative 16GB estimate
    else:
        # CPU-only systems (like Mac Mini M4) - estimate system RAM available for models
        try:
            import resource
            # Get system memory via resource limit
            max_memory = resource.getrlimit(resource.RLIMIT_AS)[0]
            if max_memory != resource.RLIM_INFINITY:
                # Very conservative for CPU-only: reserve 50% for system
                available_memory = max_memory * 0.5
                return available_memory / (1024**3)
        except (ImportError, OSError):
            pass
        # Fallback for CPU-only systems
        return 8.0  # Conservative 8GB estimate for CPU-only

def get_available_gpu_memory():
    """Returns currently available memory for models in GiB."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_memory, total_memory = torch.cuda.mem_get_info()
        return free_memory / (1024**3)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # For MPS, estimate based on conservative fallback
        # Without psutil, use conservative estimate
        return 8.0  # Conservative 8GB estimate for available memory
    else:
        # CPU-only systems - very conservative estimate
        return 4.0  # Conservative 4GB estimate for CPU-only available memory

def clear_gpu_cache():
    """Clear GPU memory cache for both CUDA and MPS."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()

def is_memory_pressure():
    """Check if system is under memory pressure (simplified without psutil)."""
    # Without psutil, use torch memory info as proxy
    try:
        if torch.cuda.is_available():
            free_memory, total_memory = torch.cuda.mem_get_info()
            # Consider pressure if less than 2GB GPU memory available
            return free_memory < 2 * (1024**3)
        else:
            # For CPU-only systems (like Mac Mini M4), be more conservative
            # Use a simple heuristic based on available memory estimate
            available = get_available_gpu_memory()
            # Consider pressure if less than 2GB estimated available
            return available < 2.0
    except Exception:
        return False

def gpu_model_choice_resolver(large_model_pool, small_model_pool):
    """Selects a model from the large or small pool based on VRAM."""
    vram = get_gpu_vram()
    available_vram = get_available_gpu_memory()
    
    # Use more conservative thresholds for Apple Silicon
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon: be more conservative due to unified memory
        if vram >= 32 and available_vram >= 16 and not is_memory_pressure():
            model_pool = large_model_pool
        else:
            model_pool = small_model_pool
    else:
        # CUDA: original logic
        if vram >= 40:
            model_pool = large_model_pool
        else:
            model_pool = small_model_pool
    
    return random.choice(model_pool)

OmegaConf.register_new_resolver("gpu_model_choice", gpu_model_choice_resolver)
