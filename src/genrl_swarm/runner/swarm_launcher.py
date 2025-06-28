import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from genrl_swarm.communication.communication import Communication
from genrl_swarm.communication.hivemind.hivemind_backend import \
    HivemindBackend, HivemindRendezvouz

from genrl_swarm.misc_utils.omega_gpu_resolver import gpu_model_choice_resolver # necessary for gpu_model_choice resolver in hydra config

@hydra.main(version_base=None)
def main(cfg: DictConfig):
    is_master=False
    HivemindRendezvouz.init(is_master=is_master)    

    game_manager = instantiate(cfg.game_manager)
    game_manager.run_game()


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    Communication.set_backend(HivemindBackend)
    main()
