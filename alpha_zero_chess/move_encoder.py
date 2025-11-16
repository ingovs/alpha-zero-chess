import chess

# --- Spatial Move Representation for Chess ---
# The policy output is a stack of 73 planes of size 8x8.
# Each plane represents a different type of move from a given square.

# 56 planes for queen-like moves (rooks, bishops, queens)
#  - Each of the 8 directions (N, NE, E, SE, S, SW, W, NW)
#  - For each direction, 7 possible move distances
# 8 planes for knight moves
# 9 planes for pawn underpromotions (to Rook, Bishop, or Knight)
#  - 3 directions for promotion captures (left, forward, right)
#  - 3 promotion piece types (R, B, N)
NUM_MOVE_PLANES = 73

# --- Directional constants ---
DIRECTIONS = {
    'N': (0, 1), 'NE': (1, 1), 'E': (1, 0), 'SE': (1, -1),
    'S': (0, -1), 'SW': (-1, -1), 'W': (-1, 0), 'NW': (-1, 1)
}
KNIGHT_MOVES = [
    (1, 2), (1, -2), (-1, 2), (-1, -2),
    (2, 1), (2, -1), (-2, 1), (-2, -1)
]

# --- Action to Move Mapping ---
ACTION_TO_MOVE_MAP = {}
MOVE_TO_ACTION_MAP = {}

def _initialize_move_maps():
    plane_idx = 0

    # 1. Queen-like moves (56 planes)
    for direction, (df, dr) in DIRECTIONS.items():
        for dist in range(1, 8):
            ACTION_TO_MOVE_MAP[plane_idx] = {'type': 'queen', 'dr': dr, 'df': df, 'dist': dist}
            plane_idx += 1

    # 2. Knight moves (8 planes)
    for df, dr in KNIGHT_MOVES:
        ACTION_TO_MOVE_MAP[plane_idx] = {'type': 'knight', 'dr': dr, 'df': df}
        plane_idx += 1

    # 3. Underpromotions (9 planes)
    for piece in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
        for df in [-1, 0, 1]: # Capture left, forward, capture right
            ACTION_TO_MOVE_MAP[plane_idx] = {'type': 'underpromotion', 'piece': piece, 'df': df}
            plane_idx += 1

_initialize_move_maps()

def move_to_action(move: chess.Move):
    """
    Converts a chess.Move object to a tuple of (plane_index, from_square).
    """
    from_sq = move.from_square
    to_sq = move.to_square

    # --- Underpromotions ---
    if move.promotion and move.promotion != chess.QUEEN:
        df = chess.square_file(to_sq) - chess.square_file(from_sq)
        for idx, action in ACTION_TO_MOVE_MAP.items():
            if action.get('type') == 'underpromotion' and \
               action.get('piece') == move.promotion and \
               action.get('df') == df:
                return idx, from_sq

    # --- Knight moves ---
    dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)
    df = chess.square_file(to_sq) - chess.square_file(from_sq)

    for idx, action in ACTION_TO_MOVE_MAP.items():
        if action.get('type') == 'knight' and \
           action.get('dr') == dr and action.get('df') == df:
            return idx, from_sq

    # --- Queen-like moves ---
    dist = max(abs(dr), abs(df))
    if dr != 0: dr //= dist
    if df != 0: df //= dist

    for idx, action in ACTION_TO_MOVE_MAP.items():
        if action.get('type') == 'queen' and \
           action.get('dr') == dr and action.get('df') == df and \
           action.get('dist') == dist:
            return idx, from_sq

    return None

def action_to_move(board: chess.Board, action_tuple):
    """
    Converts a tuple of (plane_index, from_square) back to a chess.Move object.
    Returns the move if it is legal for the given board, otherwise None.
    """
    plane_idx, from_sq = action_tuple
    action = ACTION_TO_MOVE_MAP[plane_idx]

    from_rank = chess.square_rank(from_sq)
    from_file = chess.square_file(from_sq)

    if action['type'] in ['queen', 'knight']:
        dr, df = action['dr'], action['df']
        dist = action.get('dist', 1)

        to_rank = from_rank + dr * dist
        to_file = from_file + df * dist

        if 0 <= to_rank < 8 and 0 <= to_file < 8:
            to_sq = chess.square(to_file, to_rank)
            move = chess.Move(from_sq, to_sq)
            if move in board.legal_moves:
                return move

    elif action['type'] == 'underpromotion':
        piece = action['piece']
        df = action['df']

        # Determine promotion rank
        to_rank = 7 if board.turn == chess.WHITE else 0
        to_file = from_file + df

        if 0 <= to_file < 8:
            to_sq = chess.square(to_file, to_rank)
            move = chess.Move(from_sq, to_sq, promotion=piece)
            if move in board.legal_moves:
                return move

    return None
