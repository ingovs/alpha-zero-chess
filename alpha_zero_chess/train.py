import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from alpha_zero_chess.model import AlphaZeroNet
from alpha_zero_chess.config import BATCH_SIZE, EPOCHS, LEARNING_RATE

from alpha_zero_chess.model import AlphaZeroNet, board_to_input

class Trainer:
    def __init__(self):
        self.model = AlphaZeroNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.policy_criterion = torch.nn.KLDivLoss(reduction='batchmean')
        self.value_criterion = torch.nn.MSELoss()

    def train(self, examples):
        boards, policies, values = zip(*examples)

        states = torch.FloatTensor([board_to_input(b) for b in boards])
        policies = torch.FloatTensor(policies)
        values = torch.FloatTensor(values)

        dataset = TensorDataset(states, policies, values)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(EPOCHS):
            for batch_states, batch_policies, batch_values in dataloader:
                self.optimizer.zero_grad()

                log_ps, vs = self.model(batch_states)

                policy_loss = self.policy_criterion(log_ps, batch_policies)
                value_loss = self.value_criterion(vs.view(-1), batch_values)

                loss = policy_loss + value_loss
                loss.backward()
                self.optimizer.step()

    def save_model(self, path="alpha_zero_chess.pth"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="alpha_zero_chess.pth"):
        self.model.load_state_dict(torch.load(path))
