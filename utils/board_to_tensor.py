"""

- Input: The CNN takes in a 119-channel 8x8 image stack representing the board state, including piece positions,
turn information, and game-specific rules.
- Output: The CNN outputs an 8x8x73 stack of planes representing a probability distribution over possible moves,
categorized by starting square and move type.
"""
from typing import List

import chess
import numpy as np

# Mapping from piece type and color to tensor layer
PIECE_TO_LAYER = {
    chess.PAWN: 0,
    chess.ROOK: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

def print_tensor(tensor):
    # Print tensor layers for verification
    piece_names = [
        "White Pawns", "White Rooks", "White Knights", "White Bishops",
        "White Queen", "White King", "Black Pawns", "Black Rooks",
        "Black Knights", "Black Bishops", "Black Queen", "Black King"
    ]

    for i, name in enumerate(piece_names):
        print(f"Layer {i} ({name}):")
        print(tensor[:, :, i])


def board_to_tensor(board):
    """
    Convert a chess board to a tensor representation.

    Parameters
    ----------
    board : chess.Board
        The chess board to be converted.

    Returns
    -------
    numpy.ndarray
        A tensor of shape (8, 8, 12) representing the board. Each layer corresponds to a specific piece type and color.
        The first 6 layers represent white pieces (pawn, knight, bishop, rook, queen, king) and the next 6 layers represent black pieces.
        The tensor is initialized to zeros, and a value of 1 is set for the presence of a piece at the corresponding position.
    """
    # Create an 8x8x12 tensor initialized to zeros
    tensor = np.zeros((8, 8, 12), dtype=int)

    # Iterate over all 64 squares
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)  # Convert square index to row and column
            # each piece and its color will have its layer on the CNN
            # (e.g., white pawns = layer 0, black queen = layer 10)
            layer = PIECE_TO_LAYER[piece.piece_type] + (0 if piece.color == chess.BLACK else 6)
            tensor[row, col, layer] = 1  # Set the tensor value to 1 for this piece

    return tensor


def initialize_input_planes():
    """
    Creates the initial input planes for the beginning of a chess game.

    Converts a python-chess board to an 8x8x119 numpy array (tensor).
    This represents the board state, including piece positions, turn information, and game-specific rules.

    Args:
        board: A chess.Board object.

    Returns:
        A numpy array representing the board state as an 8x8x119 stack of planes.
    """
    # Set the starting position for the current time step (t=0)
    board = chess.Board()

    # Initialize the 8x8x119 stack
    input_planes = np.zeros((8, 8, 119), dtype=np.float32)

    # 1. Piece features (M=14, T=8) - 8x8x112
    # Previous time steps (t=-1 to t=-7) are all zeros (or default values)
    # (No need to explicitly set them as they are already initialized to zero)
    tensor = board_to_tensor(board)
    input_planes[:, :, 0:12] = tensor

    # Add the repetition planes (assuming no history for simplification)
    # There are two repetitions planes (for each position from the most recent T=8 positions):
    # a) The first repetition plane will be a plane where all the entries are 1's if the position is being repeated for the first time. Else 0's.
    # b) The second repetition plane will be a plane where all the entries are 1's if the current position is being repeated for the second time. Else 0's.
    if board.is_repetition(2):
      input_planes[:, :, 12] = 1
    if board.is_repetition(3):
      input_planes[:, :, 13] = 1

    # TODO: stopped checking here
    # 2. Additional game-specific features (L=7) - 8x8x7
    # Player's turn
    input_planes[:, :14] = int(board.turn)  # 1 for White, 0 for Black

    # Total move count (normalized)
    input_planes[:, :15] = board.fullmove_number / 100.0

    # Castling rights
    input_planes[:, :16] = board.has_kingside_castling_rights(chess.WHITE)
    input_planes[:, :17] = board.has_queenside_castling_rights(chess.WHITE)
    input_planes[:, :18] = board.has_kingside_castling_rights(chess.BLACK)
    input_planes[:, :19] = board.has_queenside_castling_rights(chess.BLACK)

    # No-progress count (normalized)
    input_planes[:, :20] = board.halfmove_clock / 100.0

    return input_planes


# TODO: review this function
def update_input_planes(prev_input_planes, board, move):
    """
    Updates the input planes after a move is made.

    Args:
        prev_input_planes: An 8x8x119 numpy array representing the previous board states.
        board: A chess.Board object representing the current board state.
        move: The chess.Move object representing the move that was just made.

    Returns:
        A new 8x8x119 numpy array representing the updated board states,
        including the new current state and the shifted history.
    """

    # 1. Shift the History
    new_input_planes = np.zeros_like(prev_input_planes)  # Initialize with zeros

    # Copy the last 7 time steps from the previous input to time steps 1-7 (14 stacks per game state)
    new_input_planes[14:112, :, :] = prev_input_planes[0:98, :, :]

    # 2. Update the New Current State (Time Step t=0)
    # Make a copy of the board and apply the move
    new_board = board.copy()
    new_board.push(move)

    # Update the piece positions based on the new board state
    new_input_planes[0:12, :, :] = 0  # Reset the piece planes for the new state
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    piece_indices = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}

    for piece_type in piece_types:
        for square in chess.SQUARES:
            piece = new_board.piece_at(square)
            if piece:
                rank, file = chess.square_rank(square), chess.square_file(square)
                if piece.color == chess.WHITE:
                    new_input_planes[piece_indices[piece_type], rank, file] = 1
                else:
                    new_input_planes[6 + piece_indices[piece_type], rank, file] = 1

    # Update repetitions
    if new_board.is_repetition(2):
        new_input_planes[12, :, :] = 1
    else:
        new_input_planes[12, :, :] = 0

    if new_board.is_repetition(3):
        new_input_planes[13, :, :] = 1
    else:
        new_input_planes[13, :, :] = 0

    # Update the player's turn
    new_input_planes[14, :, :] = int(new_board.turn)

    # Update the total move count
    new_input_planes[15, :, :] = new_board.fullmove_number / 100.0

    # Update castling rights
    new_input_planes[16, :, :] = int(new_board.has_kingside_castling_rights(chess.WHITE))
    new_input_planes[17, :, :] = int(new_board.has_queenside_castling_rights(chess.WHITE))
    new_input_planes[18, :, :] = int(new_board.has_kingside_castling_rights(chess.BLACK))
    new_input_planes[19, :, :] = int(new_board.has_queenside_castling_rights(chess.BLACK))

    # Update the no-progress count
    new_input_planes[20, :, :] = new_board.halfmove_clock / 100.0

    return new_input_planes


if __name__ == "__main__":
    # Initialize the board
    board = chess.Board()

    # Create the initial input planes
    input_planes = initialize_input_planes()

    # Make a move (e.g., e2e4)
    move = chess.Move.from_uci("e2e4")
    board.push(move)

    # Update the input planes after the move
    input_planes = update_input_planes(input_planes, board, move)

    # Make another move (e.g., c7c5)
    move = chess.Move.from_uci("c7c5")
    board.push(move)

    # Update the input planes after the second move
    input_planes = update_input_planes(input_planes, board, move)

    # ...and so on...