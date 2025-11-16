from alpha_zero_chess.train import Trainer
from alpha_zero_chess.mcts import ChessMCTS
from alpha_zero_chess.config import SAVE_MODEL_CYCLES

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    trainer = Trainer()
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
    main()
