import math
from typing import List, Optional

import chess
import numpy as np

from alpha_zero_chess.config import EXPLORATION_CONSTANT, NUM_SIMULATIONS
from alpha_zero_chess.move_encoder import move_to_action, action_to_move, ALL_POSSIBLE_MOVES

class MCTSNode:
    def __init__(
        self,
        board: chess.Board,
        move: Optional[chess.Move] = None,
        parent: Optional["MCTSNode"] = None,
        prior: float = 0.0,
    ):
        self.board = board.copy()
        self.move = move
        self.parent = parent
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.value = 0.0
        self.prior = prior

    def is_fully_expanded(self) -> bool:
        return len(self.children) > 0

    def is_terminal(self) -> bool:
        return self.board.is_game_over()

    def ucb_value(self) -> float:
        if self.visits == 0:
            q_value = 0
        else:
            q_value = self.value / self.visits

        return q_value + EXPLORATION_CONSTANT * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)

    def select_best_child(self) -> "MCTSNode":
        return max(self.children, key=lambda child: child.ucb_value())

    def expand(self, move_probs):
        for action, prob in enumerate(move_probs):
            if prob > 0:
                move_uci = ALL_POSSIBLE_MOVES[action]
                move = chess.Move.from_uci(move_uci)
                if move in self.board.legal_moves:
                    new_board = self.board.copy()
                    new_board.push(move)
                    self.children.append(MCTSNode(new_board, move, self, prob))

    def backpropagate(self, result: float):
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(-result)


class ChessMCTS:
    def __init__(self, model):
        self.model = model
        self.root = MCTSNode(chess.Board())

    def search(self, board: chess.Board):
        self.root = MCTSNode(board)

        for _ in range(NUM_SIMULATIONS):
            node = self.root
            while node.is_fully_expanded():
                node = node.select_best_child()

            if not node.is_terminal():
                policy, value = self.model.predict(node.board)
                node.expand(policy)
                node.backpropagate(value)
            else:
                outcome = node.board.outcome()
                if outcome is None:
                    value = 0.0
                elif outcome.winner == node.board.turn:
                    value = 1.0
                else:
                    value = -1.0
                node.backpropagate(value)

    def get_move_probabilities(self, board: chess.Board, temp=1.0):
        self.search(board)

        move_visits = [(child.move, child.visits) for child in self.root.children]

        if not move_visits:
            return [], []

        moves, visits = zip(*move_visits)

        move_probs = np.array(visits) ** (1/temp)
        move_probs /= np.sum(move_probs)

        return moves, move_probs

    def self_play(self):
        training_examples = []
        board = chess.Board()

        while not board.is_game_over():
            moves, move_probs = self.get_move_probabilities(board)

            if not moves:
                break

            action_probs = np.zeros(len(ALL_POSSIBLE_MOVES))
            for move, prob in zip(moves, move_probs):
                action = move_to_action(move)
                action_probs[action] = prob

            training_examples.append([board.copy(), action_probs, None])

            move = np.random.choice(moves, p=move_probs)
            board.push(move)

        # Assign game outcome to the training examples
        outcome = board.outcome()
        if outcome is None:
            result = 0.0
        elif outcome.winner == chess.WHITE:
            result = 1.0
        else: # Black wins
            result = -1.0

        for i in range(len(training_examples)):
            # The result is from the perspective of the current player at that state
            if training_examples[i][0].turn == chess.WHITE:
                training_examples[i][2] = result
            else:
                training_examples[i][2] = -result

        return training_examples
