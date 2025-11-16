import math
from typing import List, Optional

import chess
import numpy as np

from alpha_zero_chess.config import EXPLORATION_CONSTANT, NUM_SIMULATIONS
from alpha_zero_chess.move_encoder import move_to_action, action_to_move, NUM_MOVE_PLANES

class MCTSNode:
    def __init__(
        self,
        board: chess.Board,
        move: Optional[chess.Move] = None,
        parent: Optional["MCTSNode"] = None,
        prior: float = 0.0,
        history: List[chess.Board] = None
    ):
        self.board = board.copy()
        self.move = move
        self.parent = parent
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.value = 0.0
        self.prior = prior
        self.history = history if history is not None else [board.copy()]

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

    def expand(self, policy_planes):
        legal_moves = list(self.board.legal_moves)
        for move in legal_moves:
            action = move_to_action(move)
            if action:
                plane_idx, from_sq = action
                prob = policy_planes[plane_idx, from_sq // 8, from_sq % 8]

                new_board = self.board.copy()
                new_board.push(move)
                new_history = self.history + [new_board.copy()]
                if len(new_history) > 8:
                    new_history.pop(0)
                self.children.append(MCTSNode(new_board, move, self, prob, new_history))

    def backpropagate(self, result: float):
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(-result)


class ChessMCTS:
    def __init__(self, model):
        self.model = model
        self.root = MCTSNode(chess.Board())

    def search(self, board: chess.Board, history: List[chess.Board]):
        self.root = MCTSNode(board, history=history)

        for _ in range(NUM_SIMULATIONS):
            node = self.root
            while node.is_fully_expanded():
                node = node.select_best_child()

            if not node.is_terminal():
                policy, value = self.model.predict(node.history)
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

    def get_move_probabilities(self, board: chess.Board, history: List[chess.Board], temp=1.0):
        self.search(board, history)

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
        history = [board.copy()]

        while not board.is_game_over():
            moves, move_probs = self.get_move_probabilities(board, history)

            if not moves:
                break

            policy_target = np.zeros((NUM_MOVE_PLANES, 8, 8))
            for move, prob in zip(moves, move_probs):
                action = move_to_action(move)
                if action:
                    plane_idx, from_sq = action
                    policy_target[plane_idx, from_sq // 8, from_sq % 8] = prob

            training_examples.append([history, policy_target, None])

            move = np.random.choice(moves, p=move_probs)
            board.push(move)

            history.append(board.copy())
            if len(history) > 8:
                history.pop(0)

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
            if training_examples[i][0][-1].turn == chess.WHITE:
                training_examples[i][2] = result
            else:
                training_examples[i][2] = -result

        return training_examples
