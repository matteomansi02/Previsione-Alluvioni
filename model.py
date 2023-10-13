import torch
import torch.nn as nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model_preds(nn.Module):
    """MODEL WITH PREDICTIONS WITH TWO ENCODERS THAT HAVE HALF THE HIDDEN SIZE OF DECODER"""

    def __init__(self, input_size: int, hidden_size: int, n_timesteps_out: int):

        super(Model_preds, self).__init__()
        self.hidden_size = hidden_size
        self.n_timesteps_out = n_timesteps_out
        print('onlyonce', self.n_timesteps_out)
        self.input_size = input_size
        # encoder prende 72, 6 e ci teniamo l'ultimo stato 32, 1
        self.encoder1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size//2)
        #encoder prende 24, 5
        self.encoder2 = nn.LSTM(input_size=self.input_size - 1, hidden_size=self.hidden_size//2)
        # decoder prende 1, 1
        self.decoder =  nn.LSTM(input_size=1, hidden_size=self.hidden_size)
        # linear prende l'ultimo stato del decoder 32, 1
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, x: torch.Tensor, preds: torch.Tensor, batch_size: int, n_features_out) -> torch.Tensor:

        _, (h_n1, c_n1) = self.encoder1(x)
        _, (h_n2, c_n2) = self.encoder2(preds)
        states = (torch.cat([h_n1,h_n2],2), torch.cat([c_n1,c_n2],2))
        decoder_input_data = torch.zeros((1, batch_size, n_features_out)).to(DEVICE)
        #print(decoder_input_data.shape, states[0].shape)
        all_outputs = []
        for _ in range(self.n_timesteps_out):
            outputs, (state_h, state_c) = self.decoder(decoder_input_data, (states))
            # aggiorno gli stati per il prossimo ciclo (32,1)
            states = (state_h, state_c)
            # converto in (1,1)
            outputs = self.fc(outputs)
            all_outputs.append(outputs)
            decoder_input_data = outputs

        return all_outputs