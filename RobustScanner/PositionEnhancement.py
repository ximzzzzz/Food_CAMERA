import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionAwareModule(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, lstm_layers, position_direction=1):
        super(PositionAwareModule, self).__init__()
        if position_direction==1:
            self.lstm = nn.LSTM(input_size, hidden_size, lstm_layers, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size, int(hidden_size/2), lstm_layers, bidirectional=True, batch_first=True)
        
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, 
                               padding=1
                              )
        self.conv2 = nn.Conv2d(hidden_size, output_size, kernel_size=3, 
                               padding=1
                              )
        
    def forward(self, features):
        self.lstm.flatten_parameters() 
        rows = features.size(2)
        lstm_fmap = features.permute(0,2,3,1)
        lstm_row_list = []
        for i in range(rows):
            row_res, _ = self.lstm(features.permute(0, 2, 3, 1)[:, i, :, :])
            lstm_row_list.append(row_res.unsqueeze(1))
        lstm_fmap = torch.cat(lstm_row_list, dim=1)
        conv_fmap = lstm_fmap.permute(0, 3, 1, 2)
        
        conv_fmap = self.conv1(conv_fmap)
        torch.relu_(conv_fmap)
        conv_fmap = self.conv2(conv_fmap)
#         conv_fmap = conv_fmap.permute(0, 2, 3, 1)
        return conv_fmap
        
        
        

class AttnModule(nn.Module):
    def __init__(self, opt, hidden_size, num_classes, device):
        super(AttnModule, self).__init__()
#         self.context_refinement = nn.Linear(opt.output_channel, opt.hidden_size)
        self.generator = nn.Linear(opt.output_channel, num_classes)
#         self.embedding_weight = nn.Linear(hidden_size, opt.output_channel)
#         self.embedding_score = nn.Linear(hidden_size, 1)
        self.position_embedding_layer = nn.Embedding(opt.batch_max_length+1, opt.output_channel)
        self.opt = opt
        self.device = device
        
    def forward(self, position_fmap, origin_fmap): # channel last input, throught .permute(batchsize, height, width, channel)
        num_steps = self.opt.batch_max_length + 1
        batch_size = position_fmap.size(0)
        hidden_size = position_fmap.size(-1)
        output_hiddens = torch.FloatTensor(batch_size, num_steps, hidden_size).fill_(0).to(self.device)

        for i in range(num_steps):
            position_embedded = self.position_embedding_layer(torch.LongTensor(batch_size).fill_(i).to(self.device))
#             position_embedded = self.position_embedding_layer(torch.LongTensor(1).fill_(i).to(self.device)) ####### temp for deployment!
            a = torch.softmax(torch.bmm(position_fmap.view(batch_size, -1, hidden_size) , position_embedded.unsqueeze(2)), 1)
            
            ######### Bahdanau et al. (2015) ###########
#             position_weighted = self.embedding_weight(position_embedded) #
#             position_score = self.embedding_score(position_weighted)   #
#             position_attention = torch.softmax(position_score, 1)   #
#             context = torch.bmm(position_attention.permute(0,2,1), batch_h) #
            ##########################################
    
#             print('a : ', a.permute(0,2,1).shape)
#             print('origin fmap : ', origin_fmap.shape)
#             print('position fmap : ', position_fmap.shape)
#             print('origin fmap view : ' , origin_fmap.view(batch_size, -1, hidden_size).shape)
            context = torch.bmm(a.permute(0,2,1), origin_fmap.view(batch_size, -1, hidden_size))
            output_hiddens[:, i, :] = context.squeeze(1)
        g_prime = self.generator(output_hiddens)
            
        return g_prime        
    
    
class DynamicallyFusingModule(nn.Module):
    def __init__(self, n_classes):
        super(DynamicallyFusingModule , self).__init__()
        self.lin1 = nn.Linear(n_classes *2, n_classes)
        self.lin2 = nn.Linear(n_classes *2, n_classes)
    
    def forward(self, g, g_prime):
#         print('g : ', g.shape)
#         print('g_prime : ', g_prime.shape)
        concat = torch.cat([g, g_prime],dim=-1)
#         print(concat.shape)
        lin1_res = self.lin1(concat)
        torch.sigmoid_(lin1_res)
        lin2_res = self.lin2(concat)
        pred = lin1_res * lin2_res
        
        return pred