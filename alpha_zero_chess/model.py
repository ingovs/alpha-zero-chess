import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess

from alpha_zero_chess.config import INPUT_SHAPE, NUM_RESIDUAL_BLOCKS, NUM_FILTERS
from alpha_zero_chess.move_encoder import ALL_POSSIBLE_MOVES

def board_to_input(board):
    """
    Converts a chess.Board object to a numpy array suitable for the neural network.
    The input shape is (18, 8, 8).
    """
    input_board = np.zeros(INPUT_SHAPE, dtype=np.float32)

    # Piece positions
    for piece_type in chess.PIECE_TYPES:
        for square in board.pieces(piece_type, board.turn):
            idx = piece_type - 1
            input_board[idx, square // 8, square % 8] = 1
        for square in board.pieces(piece_type, not board.turn):
            idx = piece_type - 1 + 6
            input_board[idx, square // 8, square % 8] = 1

    # Repetition counters
    if board.is_repetition(2):
        input_board[12, :, :] = 1
    if board.is_repetition(3):
        input_board[13, :, :] = 1

    # Color
    if board.turn == chess.WHITE:
        input_board[14, :, :] = 1
    else:
        input_board[14, :, :] = 0

    # Total move count
    input_board[15, :, :] = board.fullmove_number

    # Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        input_board[16, 0, 0] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        input_board[16, 0, 1] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        input_board[17, 0, 0] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        input_board[17, 0, 1] = 1

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
        self.conv_policy = nn.Conv2d(NUM_FILTERS, 2, kernel_size=1, stride=1, bias=False)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2 * 8 * 8, len(ALL_POSSIBLE_MOVES))

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
        policy = self.relu(policy)
        policy = policy.view(policy.size(0), -1)
        policy = self.fc_policy(policy)
        policy = F.log_softmax(policy, dim=1)

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

    def predict(self, board):
        self.eval()
        with torch.no_grad():
            input_board = torch.FloatTensor(board_to_input(board)).unsqueeze(0)
            log_ps, v = self(input_board)

            ps = torch.exp(log_ps)

            return ps.numpy()[0], v.numpy()[0][0]
