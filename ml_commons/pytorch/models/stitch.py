from torch import nn


class StitchedModel(nn.Sequential):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            if type(module) == tuple:
                assert len(module) == 3
                module, start, end = module
                module = nn.Sequential(*list(module.children())[start:end])
            self.add_module(str(idx), module)
