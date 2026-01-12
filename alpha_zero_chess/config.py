# MCTS CONFIG
NUM_SIMULATIONS = 800
EXPLORATION_CONSTANT = 1.0
DIRICHLET_ALPHA = 0.3  # Noise concentration (0.3 for chess, 0.03 for Go)
DIRICHLET_EPSILON = 0.25  # Noise weight (25% noise, 75% prior)

# NEURAL NETWORK
INPUT_SHAPE = (119, 8, 8)
NUM_RESIDUAL_BLOCKS = 19
NUM_FILTERS = 256

# TRAINING
BATCH_SIZE = 4096
EPOCHS = 10
SAVE_MODEL_CYCLES = 10
"""
AlphaZero doesn't use a single fixed learning rate; it uses a schedule that decays the rate as training progresses.
This allows the network to learn quickly at first and then refine its weights more precisely later on.

📉Here is the specific schedule used in the paper for Chess:
Training Steps   Learning Rate
0 to 100,000    0.2
100,000 to 300,000  0.02
300,000 to 500,000  0.002
500,000+        0.0002
"""
LEARNING_RATE = 0.001
