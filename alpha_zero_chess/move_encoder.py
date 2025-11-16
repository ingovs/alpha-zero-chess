import chess

# A more robust move encoding scheme that can handle all possible moves in chess.
# The policy output of the neural network will have a size of 4672, representing
# all possible moves.

def generate_uci_moves():
    moves = []
    for from_square in chess.SQUARES:
        for to_square in chess.SQUARES:
            # Pawn promotions
            if (chess.square_rank(from_square) == 6 and chess.square_rank(to_square) == 7) or \
               (chess.square_rank(from_square) == 1 and chess.square_rank(to_square) == 0):
                for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    moves.append(chess.Move(from_square, to_square, promotion=promotion).uci())
            else:
                moves.append(chess.Move(from_square, to_square).uci())
    return sorted(list(set(moves)))

ALL_POSSIBLE_MOVES = generate_uci_moves()
MOVE_TO_ACTION = {move: i for i, move in enumerate(ALL_POSSIBLE_MOVES)}
ACTION_TO_MOVE = {i: move for i, move in enumerate(ALL_POSSIBLE_MOVES)}

def move_to_action(move: chess.Move) -> int:
    """Converts a chess.Move object to a numerical action."""
    return MOVE_TO_ACTION.get(move.uci())

def action_to_move(action: int) -> chess.Move:
    """Converts a numerical action back to a chess.Move object."""
    return chess.Move.from_uci(ACTION_TO_MOVE.get(action))
