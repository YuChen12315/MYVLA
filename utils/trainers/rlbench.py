import torch

from .base import BaseTrainTester


class RLBenchTrainTester(BaseTrainTester):

    @torch.no_grad()
    def prepare_batch(self, batch, augment=False):
        batch["action"] = self.preprocessor.process_actions(batch["action"])
        batch["proprioception"] = self.preprocessor.process_proprio(batch["proprioception"])
        batch["rgb"] = self.preprocessor.process_obs(
            batch["rgb"],
            augment=augment
        )
        return batch
