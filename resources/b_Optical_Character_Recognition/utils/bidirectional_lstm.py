import torch
from torch import nn


class BidirectionalLSTM(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int):
        """
        Класс для создания двунаправленной LSTM и линейного слоя после нее.

        Args:
            n_in: количество входных признаков
            n_hidden: количество признаков в скрытом состоянии
            n_out: количество выходных признаков
        """
        super(BidirectionalLSTM, self).__init__()
        self.hidden_size = n_hidden
        self.rnn = nn.LSTM(n_in, n_hidden, bidirectional=True)
        self.embedding = nn.Linear(n_hidden * 2, n_out)
        self.init_hidden_state = torch.nn.Parameter(torch.zeros((2, 1, self.hidden_size)), requires_grad=False)
        self.init_cell_state = torch.nn.Parameter(torch.zeros((2, 1, self.hidden_size)), requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size = input.size(1)
        init_hidden_state = self.init_hidden_state.expand(-1, batch_size, -1).contiguous()
        init_cell_state = self.init_cell_state.expand(-1, batch_size, -1).contiguous()

        recurrent, _ = self.rnn(input, (init_hidden_state, init_cell_state))
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output
