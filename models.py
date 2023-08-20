import torch
import torch.nn as nn

from training import get_target_device


class SimpleFNN(nn.Module):

    def __init__(
            self,
            n_input_features: int,
            n_hidden_features: int,
            n_hidden_layers: int,
            n_output_features: int,
            activation_function: nn.Module = nn.ReLU()
    ):
        super().__init__()

        self.layers = nn.Sequential()
        for i in range(n_hidden_layers):
            self.layers.add_module(f"layer_{i}", nn.Linear(n_input_features, n_hidden_features))
            self.layers.add_module(f"activation_{i}", activation_function)
            n_input_features = n_hidden_features

        self.layers.add_module(f"output_layer", nn.Linear(n_input_features, n_output_features))

    def forward(self, x: torch.Tensor):
        output = self.layers(x)
        return output


class SimpleCNN(nn.Module):

    def __init__(
            self,
            n_input_channels: int,
            n_hidden_kernels: int,
            n_hidden_layers: int,
            n_output_channels: int,
            kernel_size: int | tuple[int, int] = 3,
            stride: int | tuple[int, int] = 1,
            padding: str | int | tuple[int, int] = 0,
            padding_mode: str = "zeros",
            activation_function: nn.Module = nn.ReLU()
    ):
        super().__init__()

        self.layers = nn.Sequential()
        for i in range(n_hidden_layers):
            self.layers.add_module(
                f"layer_{i}",
                nn.Conv2d(in_channels=n_input_channels,
                          out_channels=n_hidden_kernels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          padding_mode=padding_mode)
            )
            self.layers.add_module(f"activation_{i}", activation_function)
            n_input_channels = n_hidden_kernels

        self.layers.add_module(
            f"output_layer",
            nn.Conv2d(in_channels=n_input_channels,
                      out_channels=n_output_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      padding_mode=padding_mode)
        )

    def forward(self, x: torch.Tensor):
        output = self.layers(x)
        return output


class SimpleRNN(nn.Module):

    def __init__(
            self,
            n_input_features: int,
            n_hidden_units: int,
            n_layers: int,
            n_output_features: int,
            non_linearity: str = "tanh"
    ):
        super().__init__()

        self.device = get_target_device()
        self.n_hidden_units = n_hidden_units
        self.n_layers = n_layers

        # batch_first = True ensures that the data has the following shape -> (batch_size, seq_length, input_size)
        self.rnn = nn.RNN(n_input_features, n_hidden_units, n_layers, non_linearity, batch_first=True)
        self.fc = nn.Linear(n_hidden_units, n_output_features)

    def forward(self, x: torch.Tensor):
        h0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden_units).to(self.device)

        # out has the following shape -> (batch_size, seq_length, hidden_size)
        out, _ = self.rnn(x, h0)
        # in this case we only use the last hidden state for the fully connected layer
        out = self.fc(out[:, -1, :])

        return out


class SimpleLSTM(nn.Module):

    def __init__(
            self,
            n_input_features: int,
            n_hidden_units: int,
            n_layers: int,
            n_output_features: int,
            non_linearity: str = "tanh"
    ):
        super().__init__()

        self.device = get_target_device()
        self.n_hidden_units = n_hidden_units
        self.n_layers = n_layers

        # batch_first = True ensures that the data has the following shape -> (batch_size, seq_length, input_size)
        self.rnn = nn.LSTM(n_input_features, n_hidden_units, n_layers, non_linearity, batch_first=True)
        self.fc = nn.Linear(n_hidden_units, n_output_features)

    def forward(self, x: torch.Tensor):
        h0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden_units).to(self.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden_units).to(self.device)

        # out has the following shape -> (batch_size, seq_length, hidden_size)
        out, _ = self.rnn(x, (h0, c0))
        # in this case we only use the last hidden state for the fully connected layer
        out = self.fc(out[:, -1, :])

        return out
