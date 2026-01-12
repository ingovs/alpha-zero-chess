from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess

from alpha_zero_chess.config import INPUT_SHAPE, NUM_RESIDUAL_BLOCKS, NUM_FILTERS
from alpha_zero_chess.move_encoder import NUM_MOVE_PLANES


def board_to_input(board_history: List[chess.Board]):
    """
    Converts a history of chess.Board objects to a numpy array suitable for the neural network,
    based on the AlphaZero paper's 119-plane input representation.
    """
    input_board = np.zeros(INPUT_SHAPE, dtype=np.float32)

    # --- History Features (112 planes) ---
    for i, board in enumerate(board_history):
        # Player 1 is the current player for that historical board state
        p1_color = board.turn
        p2_color = not p1_color

        # P1 and P2 piece positions (12 planes per history step)
        for piece_type in chess.PIECE_TYPES:
            # P1 pieces (6 planes)
            for square in board.pieces(piece_type, p1_color):
                plane_idx = i * 14 + (piece_type - 1)
                input_board[plane_idx, square // 8, square % 8] = 1
            # P2 pieces (6 planes)
            for square in board.pieces(piece_type, p2_color):
                plane_idx = i * 14 + 6 + (piece_type - 1)
                input_board[plane_idx, square // 8, square % 8] = 1

        # Repetition planes (2 planes per history step)
        if board.is_repetition(2):
             input_board[i * 14 + 12, :, :] = 1
        if board.is_repetition(3):
             input_board[i * 14 + 13, :, :] = 1

    # --- Constant Planes (7 planes for the current board) ---
    current_board = board_history[-1]
    p1_color = current_board.turn

    # Colour plane (plane 112)
    input_board[112, :, :] = 1 if p1_color == chess.WHITE else 0

    # Total move count plane (plane 113)
    input_board[113, :, :] = current_board.fullmove_number

    # P1 castling rights (planes 114, 115)
    if current_board.has_kingside_castling_rights(p1_color):
        input_board[114, :, :] = 1
    if current_board.has_queenside_castling_rights(p1_color):
        input_board[115, :, :] = 1

    # P2 castling rights (planes 116, 117)
    if current_board.has_kingside_castling_rights(not p1_color):
        input_board[116, :, :] = 1
    if current_board.has_queenside_castling_rights(not p1_color):
        input_board[117, :, :] = 1

    # No-progress count plane (plane 118)
    input_board[118, :, :] = current_board.halfmove_clock

    return input_board


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class AlphaZeroNet(nn.Module):
    def __init__(self):
        super(AlphaZeroNet, self).__init__()
        self.conv_input = nn.Conv2d(INPUT_SHAPE[0], NUM_FILTERS, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(NUM_FILTERS)
        self.relu = nn.ReLU(inplace=True)

        self.residual_blocks = nn.ModuleList([ResidualBlock(NUM_FILTERS, NUM_FILTERS) for _ in range(NUM_RESIDUAL_BLOCKS)])

        # Policy head
        self.conv_policy = nn.Conv2d(NUM_FILTERS, NUM_MOVE_PLANES, kernel_size=1, stride=1, bias=False)
        self.bn_policy = nn.BatchNorm2d(NUM_MOVE_PLANES)

        # Value head
        self.conv_value = nn.Conv2d(NUM_FILTERS, 1, kernel_size=1, stride=1, bias=False)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(1 * 8 * 8, 256)
        self.fc_value2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = self.relu(x)

        for block in self.residual_blocks:
            x = block(x)

        # Policy head
        policy = self.conv_policy(x)
        policy = self.bn_policy(policy)

        # Value head
        value = self.conv_value(x)
        value = self.bn_value(value)
        value = self.relu(value)
        value = value.view(value.size(0), -1)
        value = self.fc_value1(value)
        value = self.relu(value)
        value = self.fc_value2(value)
        value = torch.tanh(value)

        return policy, value

    def predict(self, board_history: List[chess.Board]):
        """
        When you call self(input_board), Python invokes the __call__ method inherited from nn.Module.

        PyTorch's nn.Module.__call__ method internally calls your forward method, plus handles
        additional functionality like:
        - Running registered hooks (pre-forward and post-forward)
        - Ensuring proper module state

        So self(input_board) is equivalent to self.forward(input_board), but using self() is the
        recommended pattern in PyTorch because it ensures all hooks and internal bookkeeping are executed properly.
        """
        # Set the model to evaluation mode (self.eval())
        self.eval()

        # Disable gradient calculation for inference (torch.no_grad()) of next move probabilities and game output value
        with torch.no_grad():
            input_board = torch.FloatTensor(board_to_input(board_history)).unsqueeze(0)

            # policy tensor and value tensor prediction by the NN
            p, v = self(input_board)

            # converts the outputs to numpy arrays
            return p.numpy()[0], v.numpy()[0][0]
