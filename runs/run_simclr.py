import hydra
from omegaconf import DictConfig, OmegaConf

from stable_ssl.config import (
    TrainerConfig,
    OptimConfig,
    ModelConfig,
    HardwareConfig,
    LogConfig,
    DataConfig,
)
from stable_ssl.trainer import Trainer
from stable_ssl.ssl_modules import SimCLR


@hydra.main(config_path="inputs", config_name="config")
def main(cfg: DictConfig):

    # Convert hydra config file to dictionary
    cfg_dict = OmegaConf.to_object(cfg)

    # Create the input for trainer
    args = TrainerConfig(
        data=DataConfig(**cfg_dict.get("data", {})),
        optim=OptimConfig(**cfg_dict.get("optim", {})),
        model=ModelConfig(**cfg_dict.get("model", {})),
        hardware=HardwareConfig(**cfg_dict.get("hardware", {})),
        log=LogConfig(**cfg_dict.get("log", {})),
    )

    # Create a trainer object
    trainer = SimCLR(args)
    trainer()


if __name__ == "__main__":
    main()