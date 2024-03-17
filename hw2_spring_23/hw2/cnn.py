import torch
import torch.nn as nn
import itertools as it
from torch import Tensor
from typing import Sequence

from .mlp import MLP, ACTIVATIONS, ACTIVATION_DEFAULT_KWARGS

POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class CNN(nn.Module):
    """
    A simple convolutional neural network model based on PyTorch nn.Modules.

    Has a convolutional part at the beginning and an MLP at the end.
    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: Sequence[int],
        pool_every: int,
        hidden_dims: Sequence[int],
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels # list of N of output channels
        self.pool_every = pool_every #
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError(f"Unsupported activation or pooling type {activation_type,pooling_type}")

        self.feature_extractor = self._make_feature_extractor()
        self.mlp = self._make_mlp()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        # ====== YOUR CODE: ======
        # Jack:
        # C-C-C-C-C <P> C-C-C-C-C <P>
        # First conv layer: works on input image volume
        # conv1 = nn.Conv2d(in_channels=x0.shape[1], out_channels=10, padding=1, kernel_size=5, stride=1)
        N = len(self.channels)
        channels_length = [in_channels] + self.channels
        for i in range(1,N+1):
            # Calc Channels length:
            in_channels = channels_length[i-1]
            out_channels = channels_length[i]
            # Conv Layer:
            conv_layer = nn.Conv2d(in_channels=in_channels, out_channels = out_channels ,**self.conv_params)
            layers.append(conv_layer)
            # Activation layer:
            # activation_layer = ACTIVATIONS[self.activation_type](**ACTIVATION_DEFAULT_KWARGS[self.activation_type],**self.activation_params)
            activation_layer = ACTIVATIONS[self.activation_type](**self.activation_params)

            layers.append(activation_layer)
            # Pooling Layer:
            if i % self.pool_every ==0:
                # Insert Pooling Layer
                pooling_layer = POOLINGS[self.pooling_type](**self.pooling_params)
                layers.append(pooling_layer)
                # Need New Calc of the the dimentions ?

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            # ====== YOUR CODE: ======
            single_pic_dim=(1,) + self.in_size
            pic = torch.rand(single_pic_dim)
            z = self.feature_extractor(pic)
            return z.numel()
            # ========================
        finally:
            torch.set_rng_state(rng_state)

    def _make_mlp(self):
        # TODO:
        #  - Create the MLP part of the model: (FC -> ACT)*M -> Linear
        #  - Use the the MLP implementation from Part 1.
        #  - The first Linear layer should have an input dim of equal to the number of
        #    convolutional features extracted by the convolutional layers.
        #  - The last Linear layer should have an output dim of out_classes.
        mlp: MLP = None
        # ====== YOUR CODE: ======
        # mlp = layers.MLP(in_features, num_classes, hidden_features=[100, 50, 100])
        in_features = self._n_features()
        num_classes = self.out_classes
        hidden_features = self.hidden_dims
        dims= self.hidden_dims + [self.out_classes]
        # Create non_lins:
        N_nonlins = len(self.hidden_dims)
        nonlins = [ACTIVATIONS[self.activation_type](**self.activation_params)]*N_nonlins + ["none"]
        mlp = MLP(in_features, dims, nonlins)
        # ========================
        return mlp

    def forward(self, x: Tensor):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        out: Tensor = None
        # ====== YOUR CODE: ======
        out_conv = self.feature_extractor(x)
        out = self.mlp(out_conv.flatten(start_dim=1)) # start_dim important ?
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))
        print(activation_type)

        if activation_type not in ACTIVATIONS:
            print(f"activation type input was {activation_type}")
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======
        N = len(channels)
        layers =[]
        channels_length = [in_channels] + channels
        for i in range(1, N + 1):
            # Calc Channels length:
            # in_channels = channels_length[i - 1] # This is the same name as the __init__
            # out_channels = channels_length[i]
            F = kernel_sizes[i-1] # kernel size
            conv_layer = nn.Conv2d(in_channels=channels_length[i - 1], out_channels=channels_length[i],kernel_size=F, padding=int(0.5*(F-1)),bias = True )
            layers.append(conv_layer)
            if i !=N:
                # Dropout abd BatchNorm:
                if dropout != 0:
                    layers.append(nn.Dropout2d(dropout))
                if batchnorm:
                    layers.append(nn.BatchNorm2d(channels_length[i]))
                # activation_layer = ACTIVATIONS[self.activation_type](**ACTIVATION_DEFAULT_KWARGS[self.activation_type],**self.activation_params)
                activation_layer = ACTIVATIONS[activation_type](**activation_params)
                layers.append(activation_layer)

        layers_short_cut=[]
        if channels[-1] != in_channels:
            short_conv_layer = nn.Conv2d(in_channels,channels[-1],kernel_size=1, bias=False)
            layers_short_cut.append(short_conv_layer)
        else:
            layers_short_cut.append(nn.Identity())
        self.main_path = nn.Sequential(*layers)
        self.shortcut_path = nn.Sequential(*layers_short_cut)


        # ========================

        # ========================

    def forward(self, x: Tensor):
        # TODO: Implement the forward pass. Save the main and residual path to `out`.
        out: Tensor = None
        # ====== YOUR CODE: ======
        out = self.main_path(x)
        # print("main_path_completed!")
        out += self.shortcut_path(x)
        # ========================
        out = torch.relu(out)
        return out


class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
        self,
        in_out_channels: int,
        inner_channels: Sequence[int],
        inner_kernel_sizes: Sequence[int],
        **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. NOT the outer projections)
            The length determines the number of convolutions, EXCLUDING the
            block input and output convolutions.
            For example, if in_out_channels=10 and inner_channels=[5],
            the block will have three convolutions, with channels 10->5->5->10.
            The first and last arrows are the 1X1 projection convolutions, 
            and the middle one is the inner convolution (corresponding to the kernel size listed in "inner kernel sizes").
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        assert len(inner_channels) > 0
        assert len(inner_channels) == len(inner_kernel_sizes)

        # TODO:
        #  Initialize the base class in the right way to produce the bottleneck block
        #  architecture.
        # ====== YOUR CODE: ======
        layers=[]
        layers_shortcut = []
        kernel_list = [1] + list(inner_kernel_sizes) + [1]
        channel_list = [inner_channels[0]] + list(inner_channels) +[in_out_channels]
        super().__init__(in_channels=in_out_channels,channels=channel_list,kernel_sizes=kernel_list,**kwargs)

        # ========================


class ResNet(CNN):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        bottleneck: bool = False,
        **kwargs,
    ):
        """
        See arguments of CNN & ResidualBlock.
        :param bottleneck: Whether to use a ResidualBottleneckBlock to group together
            pool_every convolutions, instead of a ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.bottleneck = bottleneck
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions (make sure to use the right stride and padding).
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        #  - Use bottleneck blocks if requested and if the number of input and output
        #    channels match for each group of P convolutions.
        #    Reminder: the number of convolutions performed in the bottleneck block is:
        #    2 + len(inner_channels). [1 for each 1X1 proection convolution] + [# inner convolutions].
        # - Use batchnorm and dropout as requested.
        # ====== YOUR CODE: ======
        # Jack : [(Conv->Act)*P -> Pool with skip ] * (N//P) + [Conv -> Act ] * ( N % P )
        N = len(self.channels)
        P = self.pool_every
        channels_list = [in_channels] + list(self.channels)
        # kernel_size_list = [3] * (P)
        Num_of_blocks = N//P
        for i in range(0 , Num_of_blocks ):
            # init_index = P * i
            # final_index = init_index + P -1
            # conv_channel_list = channels_list[init_index+1:final_index]

            conv_channel_list = channels_list[i*P+1 : P*(i+1)+1]

            print(conv_channel_list)
            if self.bottleneck and channels_list[i*P]==channels_list[(i+1)*P]:
                conv_channel_list = channels_list[i * P + 2: P * (i + 1) ]
                kernel_size_list = [3] * len(conv_channel_list)
                block = ResidualBottleneckBlock(in_out_channels=channels_list[i*P+1],inner_channels=conv_channel_list,
                                                inner_kernel_sizes=kernel_size_list,batchnorm=self.batchnorm,dropout=self.dropout
                                                ,activation_type=self.activation_type,activation_params=self.activation_params)
            else:
                kernel_size_list = [3] * P
                block = ResidualBlock(channels_list[i*P], conv_channel_list,kernel_size_list, self.batchnorm,self.dropout,self.activation_type,self.activation_params )
            layers.append(block)
            # Insert Pooling Layer:
            pooling_layer = POOLINGS[self.pooling_type](**self.pooling_params)
            layers.append(pooling_layer)


        print("First Step Completed! ")

        # Additional Layers
        # Residual Block:
        if N % P != 0 :
            # Need One more Residual Block:
            conv_channel_list = channels_list[1+Num_of_blocks*P:]
            kernel_size_list = [3] * len(conv_channel_list)
            block = ResidualBlock(channels_list[Num_of_blocks * P], conv_channel_list, kernel_size_list, self.batchnorm,
                                  self.dropout, self.activation_type, self.activation_params)
            layers.append(block)



        # for ii in range(N // P,N):
        #     # F = kernel_sizes[i - 1]  # kernel size
        #     conv_layer = nn.Conv2d(in_channels=channels_list[ii - 1], out_channels=channels_list[ii], kernel_size=3,
        #                            padding=int((3 - 1)//2), bias=True)
        #     layers.append(conv_layer)
        #     if True:
        #         # Dropout abd BatchNorm:
        #         if self.dropout != 0:
        #             layers.append(nn.Dropout2d(self.dropout))
        #         if self.batchnorm:
        #             layers.append(nn.BatchNorm2d(channels_list[ii]))
        #         # activation_layer = ACTIVATIONS[self.activation_type](**ACTIVATION_DEFAULT_KWARGS[self.activation_type],**self.activation_params)
        #         activation_layer = ACTIVATIONS[self.activation_type](**self.activation_params)
        #         layers.append(activation_layer)

        # ========================
        seq = nn.Sequential(*layers)
        return seq

