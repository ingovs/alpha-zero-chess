import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from alpha_zero_chess.model import AlphaZeroNet, board_to_input
from alpha_zero_chess.config import BATCH_SIZE, EPOCHS, LEARNING_RATE


class Trainer:
    def __init__(self):
        self.model = AlphaZeroNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        self.value_criterion = torch.nn.MSELoss()

    def train(self, examples):
        histories, policies, values = zip(*examples)

        states = torch.FloatTensor([board_to_input(h) for h in histories])
        policies = torch.FloatTensor(policies)
        values = torch.FloatTensor(values)

        dataset = TensorDataset(states, policies, values)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(EPOCHS):
            for batch_states, batch_policies, batch_values in dataloader:
                self.optimizer.zero_grad()

                pred_policies, pred_values = self.model(batch_states)

                # Flatten policy tensors
                pred_policies = pred_policies.view(pred_policies.size(0), -1)
                batch_policies = batch_policies.view(batch_policies.size(0), -1)

                # Policy loss: cross-entropy with soft targets (probability distributions)
                log_probs = F.log_softmax(pred_policies, dim=1)
                policy_loss = -torch.sum(batch_policies * log_probs) / batch_policies.size(0)

                value_loss = self.value_criterion(pred_values.view(-1), batch_values)

                loss = policy_loss + value_loss
                loss.backward()
                self.optimizer.step()

    def save_model(self, path="alpha_zero_chess.pth"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="alpha_zero_chess.pth"):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, weights_only=True))
            return True
        return False
