import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(torch.nn.Module):
    def __init__(self,input_size, hidden_size, num_grulstm_layers, batch_size):
        super(EncoderRNN, self).__init__()  
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_grulstm_layers = num_grulstm_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)

    def forward(self, input, hidden): # input [batch_size, length T, dimensionality d]

        output, hidden = self.gru(input, hidden)      
        return output, hidden
    
    def init_hidden(self,batch_size, device):
        #[num_layers*num_directions,batch,hidden_size]   
        return torch.zeros(self.num_grulstm_layers, batch_size, self.hidden_size, device=device)
    
class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_grulstm_layers,fc_units, output_size):
        super(DecoderRNN, self).__init__()      
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size, fc_units)
        self.out = nn.Linear(fc_units, output_size)         
        
    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden) 
        output = F.relu( self.fc(output) )
        output = self.out(output)      
        return output, hidden
    
class NetGRU(nn.Module):
    def __init__(self, input_size=118, hidden_size=1024, target_length=1, batch_size=16):
        super(NetGRU, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.lr = 0.001
        self.loss_fun = nn.MSELoss()

        self.encoder = EncoderRNN(input_size=input_size, hidden_size=hidden_size, num_grulstm_layers=2, batch_size=batch_size).to(self.device)
        self.decoder = DecoderRNN(input_size=input_size, hidden_size=hidden_size, num_grulstm_layers=2, fc_units=16, output_size=1).to(self.device)
        self.target_length = target_length


        
    def forward(self, x):
        batch_size = x.shape[0]
        # x2 = copy.deepcopy(x)
        # r = self.batch_size % batch
        # if r > 0:
        #     m = self.batch_size // batch
        #     for i in range(m-1):
        #         x = torch.cat((x, x2), dim=0)
        #     x = torch.cat((x, x2[0:r]), dim=0)
        #     print(f'after expandion x shape {x.shape}')


        input_length  = x.shape[1]
        encoder_hidden = self.encoder.init_hidden(batch_size, self.device)
        for ei in range(input_length):
            # print(f'ei = {ei}, xei shape = {x[:,ei:ei+1,:].shape}, batch is {batch}')

            encoder_output, encoder_hidden = self.encoder(x[:,ei:ei+1,:]  , encoder_hidden)
            
        decoder_input = x[:,-1,:].unsqueeze(1) # first decoder input= last element of input sequence
        decoder_hidden = encoder_hidden
        
        outputs = torch.zeros([x.shape[0], self.target_length, x.shape[2]]  ).to(self.device)
        for di in range(self.target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_input = decoder_output
            outputs[:,di:di+1,:] = decoder_output
        if self.target_length == 1:
            outputs = outputs[:, :, 0:1]
            outputs = outputs.squeeze(1)
        return outputs