# alpha-zero-chess ♟️
An implementation of AlphaZero for Chess, built for learning.

alpha-zero-chess is an educational project that reconstructs the AlphaZero algorithm using Python, PyTorch, and NumPy.

The goal is to learn the underlying mechanics behind the model described in the paper and to provide a well documented code that helps the understanding of the process.

In the end, there will be a script for testing the model vs a player with random moves.

# 🧠 How It Works
The engine learns the game solely by playing against itself, starting with zero knowledge beyond the rules of chess.

The Brain (PyTorch): A deep neural network that predicts the best moves (Policy) and estimates who is winning (Value).

The Intuition (MCTS): A search algorithm that explores future possibilities, guided by the neural network's predictions.

The Grind (Self-Play): The system plays thousands of games against itself, generating its own training data to refine the network.

Continuous Evolution: The trained model is stored in the repository, making training fully resumable. You can stop the process and pick it up later—the model's accuracy and playing strength will continue to improve as you invest more time in training.

AlphaZero:
- deep neural networks
- a general-purpose reinforcement learning algorithm
- a general-purpose tree search algorithm
