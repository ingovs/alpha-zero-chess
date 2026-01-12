from alpha_zero_chess.trainer import Trainer
from alpha_zero_chess.mcts import ChessMCTS
from alpha_zero_chess.config import SAVE_MODEL_CYCLES

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_neural_network():
    """
    Run the AlphaZero training loop for chess.

    This function initializes the trainer and MCTS, then runs self-play
    cycles to generate training examples. After each cycle, the model is
    trained on all accumulated examples and saved to disk.

    If a saved model exists, it will be loaded to continue training
    from where it left off.
    """
    trainer = Trainer()

    # Load existing model if available
    if trainer.load_model():
        logging.info("Loaded existing model, continuing training")
    else:
        logging.info("No existing model found, starting fresh")

    # NOTE: to inspect model parameters so far, uncomment the following lines:
    # # Iterator of all parameters
    # for param in trainer.model.parameters():
    #     print(param.shape, param.data)

    # # Named parameters (with layer names)
    # for name, param in trainer.model.named_parameters():
    #     print(name, param.shape)

    mcts = ChessMCTS(trainer.model)

    training_examples = []

    for i in range(1, SAVE_MODEL_CYCLES + 1):
        logging.info(f"Starting self-play cycle {i}/{SAVE_MODEL_CYCLES}")

        new_examples = mcts.self_play()
        training_examples.extend(new_examples)

        logging.info(f"Training the model with {len(training_examples)} examples")
        trainer.train(training_examples)

        logging.info("Saving the model")
        trainer.save_model()


if __name__ == "__main__":
    train_neural_network()
