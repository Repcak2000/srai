"""
Trajectory model module.

This module contains implementation of base model of trajectory.
"""

import torch
import torch.nn as nn


class TravelTimePredictionBaseModel(nn.Module):  # type: ignore
    """
    Travel time prediction base model.

    Definition of travel time prediction model
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        """
        Initialization of travel time prediction module.

        Args:
            input_size: number of input features
            hidden_size:  number of features in the hidden state of the LSTM
            num_layers: The number of recurrent layers in the LSTM
            output_size: number of output features
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_size), \
             representing the predicted travel time.
        """
        hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        cell_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_output, _ = self.lstm(x, (hidden_state, cell_state))
        fully_connected_output = self.fc(lstm_output[:, -1, :])
        return fully_connected_output
