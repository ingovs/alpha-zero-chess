import logging
import math
from typing import List, Optional

import chess
import numpy as np

from alpha_zero_chess.config import (
    EXPLORATION_CONSTANT,
    NUM_SIMULATIONS,
    DIRICHLET_ALPHA,
    DIRICHLET_EPSILON,
)
from alpha_zero_chess.move_encoder import move_to_action, action_to_move, NUM_MOVE_PLANES

logger = logging.getLogger(__name__)

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

    def ucb_value(self) -> float:
        """
        Q-Value: Exploitation
        U-Value: Exploration
        """
        if self.visits == 0:
            q_value = 0
        else:
            q_value = self.value / self.visits

        u_value = EXPLORATION_CONSTANT * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)

        return q_value + u_value

    def select_best_child(self) -> "MCTSNode":
        return max(self.children, key=lambda child: child.ucb_value())

    def expand(self, policy_planes):
        legal_moves = list(self.board.legal_moves)

        # Collect logits for legal moves
        move_logits = []
        valid_moves = []
        for move in legal_moves:
            action = move_to_action(move)
            if action:
                plane_idx, from_sq = action
                logit = policy_planes[plane_idx, from_sq // 8, from_sq % 8]
                move_logits.append(logit)
                valid_moves.append(move)

        if not valid_moves:
            return

        # Apply softmax to convert logits to probabilities
        logits = np.array(move_logits)
        logits = logits - np.max(logits)  # Numerical stability
        probs = np.exp(logits) / np.sum(np.exp(logits))

        for move, prob in zip(valid_moves, probs):
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

    def _add_dirichlet_noise(self, node: MCTSNode):
        """
        Add Dirichlet noise to the prior probabilities of the node's children.

        This encourages exploration during self-play by ensuring the agent
        doesn't always follow the network's policy exactly.
        """
        if not node.children:
            return
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(node.children))
        for child, n in zip(node.children, noise):
            child.prior = (1 - DIRICHLET_EPSILON) * child.prior + DIRICHLET_EPSILON * n

    def search(self, board: chess.Board, history: List[chess.Board]):
        self.root = MCTSNode(board, history=history)

        # Expand root node and add Dirichlet noise to encourage exploration
        policy, value = self.model.predict(self.root.history)
        self.root.expand(policy)
        self._add_dirichlet_noise(self.root)
        self.root.backpropagate(value)

        for i in range(NUM_SIMULATIONS):
            if (i + 1) % 100 == 0:
                logger.info(f"MCTS simulation {i + 1}/{NUM_SIMULATIONS}")

            node = self.root
            while node.is_fully_expanded():
                node = node.select_best_child()

            if not node.board.is_game_over():
                # policy (pi), value (v)
                policy, value = self.model.predict(node.history)
                # expand the node with the policy (NN) probabilities
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

    def get_move_probabilities(self, board: chess.Board, history: List[chess.Board], temperature: float = 1.0):
        """
        Do we go to the end of the game?
        Generally, No. This is one of the biggest differences between AlphaZero and traditional MCTS.

        In "pure" MCTS, you would play random moves all the way until someone won (a "rollout")
        to see if a path was good.

        AlphaZero is different because it uses the Value Head (v) as a crystal ball.
        - The Cut-Off: When the simulation reaches a leaf node (a state not yet visited), it stops immediately.
        - The Substitution: Instead of playing 50 more moves to see who wins, it asks the Neural Network: "Who is winning here?"
        - The Result: The network returns v (e.g., +0.8), and the algorithm treats that just like a game result, backing it up the tree.
        """
        self.search(board, history)

        move_visits = [(child.move, child.visits) for child in self.root.children]

        if not move_visits:
            return [], []

        moves, visits = zip(*move_visits)

        move_probs = np.array(visits) ** (1/temperature)
        move_probs /= np.sum(move_probs)

        return moves, move_probs

    def self_play(self):
        logger.info("Starting self-play game")
        training_examples = []
        board = chess.Board()
        history = [board.copy()]

        move_count = 0
        while not board.is_game_over():
            move_count += 1
            logger.info(f"Self-play move {move_count}")

            # Get move probabilities from MCTS after 800 simulations
            # NOTE: the game doesn't necessarily reach a terminal state within the MCTS simulations
            moves, move_probs = self.get_move_probabilities(board, history)

            if not moves:
                break

            policy_target = np.zeros((NUM_MOVE_PLANES, 8, 8))
            for move, prob in zip(moves, move_probs):
                # Converts the chess move to the AlphaZero encoding format, returning (plane_index, from_square)
                # NOTE: plane_index determines the move, in which direction and how far the piece moves (73 planes)
                # from_square is the square index (0-63) where the piece moves from
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

        logger.info("Self-play game finished with %d training examples", len(training_examples))
        return training_examples
