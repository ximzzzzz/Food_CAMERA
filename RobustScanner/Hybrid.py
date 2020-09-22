import torch
import torch.nn as nn
import torch.nn.functional as F
# device = torch.device('cuda')


class HybridBranch(nn.Module):
    
    def __init__(self, hidden_size, num_steps, num_classes, device):
        super(HybridBranch, self).__init__()
        self.embedding = torch.nn.Linear(num_classes, hidden_size)
        self.lstm = torch.nn.LSTMCell(hidden_size, hidden_size, bias=True)
        self.lstm2 = torch.nn.LSTMCell(hidden_size, hidden_size, bias=True)
        self.generator = torch.nn.Linear(hidden_size, num_classes)
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.device = device
        
    def forward(self, origin_fmap, text, is_train):
        batch_size = text.size(0)
        output_hiddens = torch.FloatTensor(batch_size, self.num_steps, self.hidden_size).fill_(0).to(self.device)
        h_t, c_t = torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(self.device), torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(self.device)
        origin_fmap_trans = origin_fmap.permute(0, 2, 3, 1).view(batch_size, -1, self.hidden_size)

        if is_train :
            for i in range(self.num_steps):
                # one-hot
                x_t = torch.FloatTensor(batch_size, self.num_classes).zero_().to(self.device)
                x_t = x_t.scatter_(1, text[:, i].unsqueeze(1), 1)
                x_t_emb = self.embedding(x_t)
#                 print(f'x_t_emb shape : {x_t_emb.shape}')
               
                h_t, c_t = self.lstm(x_t_emb, (h_t, c_t))
                h_t, c_t = self.lstm2(h_t)
#                 print('origin fmap trans : ', origin_fmap_trans.shape)
#                 print('h_t : ', h_t.unsqueeze(2).shape)
                a = torch.softmax(torch.bmm(origin_fmap_trans, h_t.unsqueeze(2)), 1)
                context = torch.bmm(a.permute(0, 2, 1), origin_fmap_trans)
                output_hiddens[:, i, :] = context.squeeze(1)
            g = self.generator(output_hiddens)

        else:
            next_input = None
            g = torch.FloatTensor(batch_size, self.num_steps, self.num_classes).fill_(0).to(self.device)

            for i in range(self.num_steps):
                if next_input==None:
                    x_t = torch.FloatTensor(batch_size, self.num_classes).zero_().to(self.device)
                    target = torch.LongTensor(batch_size, 1).fill_(0).to(self.device)
                    x_t = x_t.scatter(1, target, 1 )

                else:
                    x_t = torch.FloatTensor(batch_size, self.num_classes).zero_().to(self.device)
                    x_t = x_t.scatter_(1, next_input, 1)

                x_t_emb = self.embedding(x_t)
                a = torch.softmax(torch.bmm(origin_fmap_trans, h_t.unsqueeze(2)), 1)
                context = torch.bmm(a.permute(0, 2, 1), origin_fmap_trans)
                g_t = self.generator(context)

                _, next_input = g_t.max(2)
                g[:, i, :] = g_t.squeeze(1)

        return g