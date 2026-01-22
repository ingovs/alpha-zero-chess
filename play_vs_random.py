import random

import chess
import numpy as np

from alpha_zero_chess.mcts import ChessMCTS
from alpha_zero_chess.trainer import Trainer


def play_game(alphazero_is_white: bool = True):
    """Play a game between AlphaZero and a random player."""
    # Load the trained model
    trainer = Trainer()
    if not trainer.load_model():
        print("No trained model found. Please train a model first.")
        return

    trainer.model.eval()
    mcts = ChessMCTS(trainer.model)

    board = chess.Board()
    history = [board.copy()]
    move_count = 0

    print(f"\nStarting game: AlphaZero plays {'White' if alphazero_is_white else 'Black'}")
    print(f"Random player plays {'Black' if alphazero_is_white else 'White'}")
    print("=" * 50)
    print(board)
    print()

    while not board.is_game_over():
        move_count += 1
        is_alphazero_turn = (board.turn == chess.WHITE) == alphazero_is_white

        if is_alphazero_turn:
            # AlphaZero's turn - use MCTS
            moves, move_probs = mcts.get_move_probabilities(board, history, temperature=0.1)
            if not moves:
                print("AlphaZero has no legal moves!")
                break
            # Pick the move with highest probability (greedy)
            best_idx = np.argmax(move_probs)
            move = moves[best_idx]
            player = "AlphaZero"
        else:
            # Random player's turn
            legal_moves = list(board.legal_moves)
            move = random.choice(legal_moves)
            player = "Random"

        print(f"Move {move_count}: {player} plays {board.san(move)}")
        board.push(move)

        # Update history
        history.append(board.copy())
        if len(history) > 8:
            history.pop(0)

        print(board)
        print()

    # Game over
    print("=" * 50)
    print("Game Over!")
    outcome = board.outcome()
    if outcome is None:
        print("Result: Draw (no outcome)")
    elif outcome.winner is None:
        print(f"Result: Draw ({outcome.termination.name})")
    elif outcome.winner == chess.WHITE:
        winner = "AlphaZero" if alphazero_is_white else "Random"
        print(f"Result: White wins ({winner}) - {outcome.termination.name}")
    else:
        winner = "Random" if alphazero_is_white else "AlphaZero"
        print(f"Result: Black wins ({winner}) - {outcome.termination.name}")

    print(f"Total moves: {move_count}")


if __name__ == "__main__":
    # Play a game with AlphaZero as White
    play_game(alphazero_is_white=True)
