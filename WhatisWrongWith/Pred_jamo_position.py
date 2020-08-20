import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda')


class PositionAttnModule(nn.Module):
    def __init__(self, opt, num_classes, device):
        super(PositionAttnModule, self).__init__()
#         self.context_refinement = nn.Linear(opt.output_channel, opt.hidden_size)
        self.generator = nn.Linear(opt.hidden_size, num_classes)
        self.position_embedding_layer = nn.Embedding(opt.batch_max_length+1, opt.hidden_size)
        self.opt = opt
        self.device = device
        
    def forward(self, batch_h):
        num_steps = self.opt.batch_max_length + 1
        output_hiddens = torch.FloatTensor(self.opt.batch_size, num_steps, self.opt.hidden_size).fill_(0).to(self.device)
        
        for i in range(num_steps):
            position_embedded = self.position_embedding_layer(torch.LongTensor(self.opt.batch_size).fill_(i).to(self.device))
            a = torch.softmax(torch.bmm(batch_h, position_embedded.unsqueeze(2)), 1)
            context = torch.bmm(a.permute(0,2,1), batch_h)
            output_hiddens[:, i, :] = context.squeeze(1)
        g_prime = self.generator(output_hiddens)
            
        return g_prime

    
class DynamicallyFusingModule(nn.Module):
    def __init__(self, n_classes):
        super(DynamicallyFusingModule , self).__init__()
        self.lin1 = nn.Linear(n_classes *2, n_classes)
        self.lin2 = nn.Linear(n_classes *2, n_classes)
    
    def forward(self, g, g_prime):
        concat = torch.cat([g, g_prime],dim=-1)
        lin1_res = self.lin1(concat)
        torch.sigmoid_(lin1_res)
        lin2_res = self.lin2(concat)
        pred = lin1_res * lin2_res
        
        return pred
    
    
    
class Attention(nn.Module):
        
    def __init__(self, input_size, hidden_size, num_classes, device = device):
        super(Attention, self).__init__()
        self.device = device
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)
        
        
    def _char_to_onehot(self, input_char, onehot_dim):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(self.device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        
        return one_hot
    
    def forward(self, batch_H, text, is_train=True, batch_max_length = 25):
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1 # +1 for end of sequence
        
        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(self.device)
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(self.device),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(self.device))
        
        if is_train:
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden[0]
            g = self.generator(output_hiddens)

            
        else:

            targets = torch.FloatTensor(batch_size, self.num_classes).fill_(0).to(self.device)
            g = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(self.device)
            next_input = None
            for i in range(num_steps):
                if not next_input==None:
                    char_onehots = self._char_to_onehot(next_input, onehot_dim = self.num_classes)
                else:
                    char_onehots = targets
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden[0])

                g[:, i, :] = probs_step
                _, next_input = probs_step.max(1)

                
        return g
        
        
class AttentionCell(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size
        
    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj)) 

#         alpha = torch.sigmoid(e)
        alpha = torch.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)
        
        concat_context = torch.cat([context, char_onehots], 1)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha
    
    
class Attention_mid(nn.Module):
    
    def __init__(self, input_size, hidden_size, mid_num_classes, bot_num_classes, device = device):
        super(Attention_mid, self).__init__()
        self.device = device
        self.attention_cell = AttentionCell_mid(input_size, hidden_size, mid_num_classes, bot_num_classes)
        self.hidden_size = hidden_size
        self.mid_num_classes = mid_num_classes
        self.bot_num_classes = bot_num_classes
        self.generator = nn.Linear(hidden_size, mid_num_classes)
        
        
    def _char_to_onehot(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(self.device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        
        return one_hot
    
    def forward(self, batch_H, mid_prev_text, bot_text, is_train=True, batch_max_length = 25):
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1 # +1 for end of sequence
        
        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(self.device)
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(self.device),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(self.device))
        
        if is_train:
            for i in range(num_steps):
                char_onehots_mid = self._char_to_onehot(mid_prev_text[:, i], onehot_dim = self.mid_num_classes)
                char_onehots_bot = self._char_to_onehot(bot_text[:, i], onehot_dim = self.bot_num_classes)

                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots_mid, char_onehots_bot)
                output_hiddens[:, i, :] = hidden[0]
            g = self.generator(output_hiddens)

            
        else:
#             targets = torch.LongTensor(batch_size).fill_(0).to(self.device)
            targets = torch.FloatTensor(batch_size, self.mid_num_classes).fill_(0).to(self.device)
            g = torch.FloatTensor(batch_size, num_steps, self.mid_num_classes).fill_(0).to(self.device)
            next_input = None
            for i in range(num_steps):
                if not next_input==None:
                    char_onehots_mid = self._char_to_onehot(next_input, onehot_dim = self.mid_num_classes)
                else:
                    char_onehots_mid = targets
                    
                char_onehots_bot = self._char_to_onehot(bot_text[:, i], onehot_dim = self.bot_num_classes)                
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots_mid, char_onehots_bot)
                probs_step = self.generator(hidden[0])
                g[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
   
        return g
        
        
class AttentionCell_mid(nn.Module):
    
    def __init__(self, input_size, hidden_size, mid_num_classes, bot_num_classes):
        super(AttentionCell_mid, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + mid_num_classes + bot_num_classes, hidden_size)
        self.hidden_size = hidden_size
        
    def forward(self, prev_hidden, batch_H, char_onehots_mid, char_onehots_bot):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj)) 
        
#         alpha = torch.sigmoid(e)
        alpha = torch.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)

        concat_context = torch.cat([context, char_onehots_mid, char_onehots_bot ], 1)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha
    
    
class Attention_top(nn.Module):
    
    def __init__(self, input_size, hidden_size, top_num_classes, mid_num_classes, bot_num_classes, device = device):
        super(Attention_top, self).__init__()
        self.device = device
        self.attention_cell = AttentionCell_top(input_size, hidden_size, top_num_classes, mid_num_classes, bot_num_classes)
        self.hidden_size = hidden_size
        self.top_num_classes = top_num_classes
        self.mid_num_classes = mid_num_classes
        self.bot_num_classes = bot_num_classes
        self.generator = nn.Linear(hidden_size, top_num_classes)
        
        
    def _char_to_onehot(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(self.device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        
        return one_hot
    
    def forward(self, batch_H, top_prev_text, mid_text, bot_text, is_train=True, batch_max_length = 25):
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1 # +1 for end of sequence
        
        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(self.device)
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(self.device),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(self.device))
        
        if is_train:
            for i in range(num_steps):
                char_onehots_top = self._char_to_onehot(top_prev_text[:, i], onehot_dim = self.top_num_classes)
                char_onehots_mid = self._char_to_onehot(mid_text[:, i], onehot_dim = self.mid_num_classes)
                char_onehots_bot = self._char_to_onehot(bot_text[:, i], onehot_dim = self.bot_num_classes)

                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots_top, char_onehots_mid, char_onehots_bot)
                output_hiddens[:, i, :] = hidden[0]
            g = self.generator(output_hiddens)

            
        else:
#             targets = torch.LongTensor(batch_size).fill_(0).to(self.device)
            targets = torch.FloatTensor(batch_size, self.top_num_classes).fill_(0).to(self.device)
            g = torch.FloatTensor(batch_size, num_steps, self.top_num_classes).fill_(0).to(self.device)
            next_input = None
            for i in range(num_steps):
                if not next_input==None:
                    char_onehots_top = self._char_to_onehot(next_input, onehot_dim = self.top_num_classes)
                else:
                    char_onehots_top = targets
                    
                char_onehots_mid = self._char_to_onehot(mid_text[:, i], onehot_dim = self.mid_num_classes)
                char_onehots_bot = self._char_to_onehot(bot_text[:, i], onehot_dim = self.bot_num_classes)                    
                
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots_top, char_onehots_mid, char_onehots_bot)
                probs_step = self.generator(hidden[0])

                g[:, i, :] = probs_step
                _, next_input = probs_step.max(1)

                
        return g
        
        
class AttentionCell_top(nn.Module):
    
    def __init__(self, input_size, hidden_size, top_num_classes, mid_num_classes, bot_num_classes):
        super(AttentionCell_top, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + top_num_classes + mid_num_classes + bot_num_classes, hidden_size)
        self.hidden_size = hidden_size
        
    def forward(self, prev_hidden, batch_H, char_onehots_top, char_onehots_mid, char_onehots_bot):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj)) 
        
#         alpha = torch.sigmoid(e)
        alpha = torch.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)

        concat_context = torch.cat([context, char_onehots_top, char_onehots_mid, char_onehots_bot ], 1)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha