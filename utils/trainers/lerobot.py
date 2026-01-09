import torch

from .base import BaseTrainTester


class LerobotTrainTester(BaseTrainTester):

    @torch.no_grad()
    def prepare_batch(self, batch, augment=False):
        batch["action"] = self.preprocessor.process_actions(batch["action"])
        proprio = self.preprocessor.process_proprio(batch["proprioception"])
        rgbs, pcds = self.preprocessor.process_obs(
            batch["rgb"], batch["rgb2d"],
            batch["depth"], batch["extrinsics"], batch["intrinsics"],
            augment=augment
        )
        return (
            batch["action"],
            torch.zeros(batch["action"].shape[:-1], dtype=bool, device='cuda'),
            rgbs,
            None,
            pcds,
            batch["instr"],
            proprio
        )
