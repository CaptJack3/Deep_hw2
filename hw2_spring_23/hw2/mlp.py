import torch
from torch import Tensor, nn
from typing import Union, Sequence
from collections import defaultdict

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax,
    "lrelu": nn.LeakyReLU,
    "none": nn.Identity,
    None: nn.Identity,
}


# Default keyword arguments to pass to activation class constructors, e.g.
# activation_cls(**ACTIVATION_DEFAULT_KWARGS[name])
ACTIVATION_DEFAULT_KWARGS = defaultdict(
    dict,
    {
        ###
        "softmax": dict(dim=1),
        "logsoftmax": dict(dim=1),
    },
)


class MLP(nn.Module):
    """
    A general-purpose MLP.
    """

    def __init__(
        self, in_dim: int, dims: Sequence[int], nonlins: Sequence[Union[str, nn.Module]]
    ):
        """
        :param in_dim: Input dimension.
        :param dims: Hidden dimensions, including output dimension.
        :param nonlins: Non-linearities to apply after each one of the hidden
            dimensions.
            Can be either a sequence of strings which are keys in the ACTIVATIONS
            dict, or instances of nn.Module (e.g. an instance of nn.ReLU()).
            Length should match 'dims'.
        """
        super().__init__()
        assert len(nonlins) == len(dims)
        self.in_dim = in_dim
        self.out_dim = dims[-1]
        layers=[]
        layers =[]


        # TODO:
        #  - Initialize the layers according to the requested dimensions. Use
        #    either nn.Linear layers or create W, b tensors per layer and wrap them
        #    with nn.Parameter.
        #  - Either instantiate the activations based on their name or use the provided
        #    instances.
        # ====== YOUR CODE: ======
        L = len(nonlins)
        layer_sizes=[in_dim] + dims

        for i in range(L):
            layer = nn.Linear(layer_sizes[i],layer_sizes[i+1])
            layers.append(layer)
            if i < len(nonlins)-0:
                Act = nonlins[i]
                if  type(Act) == str:
                    layer  = ACTIVATIONS[Act](**ACTIVATION_DEFAULT_KWARGS[Act])
                else:
                    layer = Act
                layers.append(layer)
        self.fc_layers = nn.Sequential(*layers)

    def __repr__(self):
        res = "Sequential\n"
        for i,layer in enumerate(self.fc_layers):
            res += f"\t[{i}] {layer}\n"
            # return f"MLP, {self.layers}"
        return res



        # ========================

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: An input tensor, of shape (N, D) containing N samples with D features.
        :return: An output tensor of shape (N, D_out) where D_out is the output dim.
        """
        # TODO: Implement the model's forward pass. Make sure the input and output
        #  shapes are as expected.
        # ====== YOUR CODE: ======
        # Option 1:
        # for layer in self.fc_layers:
        #     x = layer(x)
        # Option 2:
        # x= torch.reshape(x,(x.shape[0],-1)) # Need ?
        x = self.fc_layers(x)
        return x
        # ========================
