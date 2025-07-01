import os
import signal
import sys
import gc
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from genrl_swarm.communication.communication import Communication
from genrl_swarm.communication.hivemind.hivemind_backend import \
    HivemindBackend, HivemindRendezvouz

from genrl_swarm.misc_utils.omega_gpu_resolver import gpu_model_choice_resolver # necessary for gpu_model_choice resolver in hydra config
from genrl_swarm.misc_utils.memory_monitor import create_memory_monitor, log_memory_stats
from genrl_swarm.logging_utils.global_defs import get_logger

def cleanup_resources():
    """Clean up resources on shutdown to prevent semaphore leaks."""
    logger = get_logger()
    logger.info("Cleaning up resources...")
    
    try:
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache if available
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        # Clean up any lingering multiprocessing resources
        import multiprocessing
        multiprocessing.active_children()  # This forces cleanup of dead children
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger = get_logger()
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    cleanup_resources()
    sys.exit(0)

@hydra.main(version_base=None)
def main(cfg: DictConfig):
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Log initial memory stats
    log_memory_stats()
    
    try:
        # Start memory monitoring
        with create_memory_monitor() as monitor:
            is_master=False
            HivemindRendezvouz.init(is_master=is_master)    

            game_manager = instantiate(cfg.game_manager)
            game_manager.run_game()
    except KeyboardInterrupt:
        get_logger().info("Interrupted by user")
    except Exception as e:
        get_logger().error(f"Unexpected error: {e}")
        raise
    finally:
        cleanup_resources()


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    Communication.set_backend(HivemindBackend)
    main()
