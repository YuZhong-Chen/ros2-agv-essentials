from pathlib import Path

import wandb
import torch.utils.tensorboard as tensorboard

class LOGGER:
    def __init__(self, project, project_name, config, project_dir, enable=True, use_wandb=True):
        self.enable = enable

        self.writer = None

        if self.enable:
            logs_dir = project_dir / "logs"
            logs_dir.mkdir(exist_ok=True)

            if use_wandb:
                wandb.init(
                    project=project,
                    name=project_name,
                    dir=project_dir,
                    sync_tensorboard=True,
                    config=config,
                )

            self.writer = tensorboard.SummaryWriter(logs_dir)
            self.writer.add_text("Hyper Parameters", "|Param|Value|\n|-|-|\n%s" % ("\n".join([f"|{param}|{value}|" for param, value in config.items()])))

            print("Save logs to", logs_dir)

    def Log(self, episode, average_loss, average_td_error, average_td_estimation, episode_reward):
        if not self.enable:
            return
        
        self.writer.add_scalar("Loss", average_loss, episode)
        self.writer.add_scalar("TD Error", average_td_error, episode)
        self.writer.add_scalar("TD Estimation", average_td_estimation, episode)
        self.writer.add_scalar("Reward", episode_reward, episode)
        self.writer.flush()