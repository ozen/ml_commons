import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as gradient_checkpoint


class CheckpointedModel(nn.Module):
    class ModelWrapper(nn.Module):
        """
        If all inputs have requires_grad=False, checkpointed model will also have requires_grad=False
        This wrapper is a workaround.
        https://github.com/pytorch/pytorch/issues/7617
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117
        """
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, x, dummy_arg):
            return self.encoder(x)

    def __init__(self, model):
        super().__init__()
        self._wrapped_model = self.ModelWrapper(model)
        self._dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def forward(self, x):
        return gradient_checkpoint(self._wrapped_model, x, self._dummy_tensor)
