from torch import nn


class StitchedModel(nn.Sequential):
    """
    Example 1:
        encoder = StitchedModel((pretrained_model, 0, -2))
    Example 2:
        classifier = StitchedModel(
            (resnet50, 7, 9),
            nn.Flatten(1),
            nn.Linear(2048, N_CLASSES)
        )
    """
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            if type(module) == tuple:
                assert len(module) == 3
                module, start, end = module
                module = nn.Sequential(*list(module.children())[start:end])
            self.add_module(str(idx), module)
