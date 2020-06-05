import torch
import torch.nn.functional as F

class CTC_decoder(torch.nn.Module):
    def __init__(self, input_size, output_size, num_classes, opt,device ):
        super(CTC_decoder, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, output_size)
        self.fcl = torch.nn.Linear(output_size, num_classes)
#         self.ctc_loss = torch.nn.CTCLoss(blank= 0, reduction='mean').to(device)
        self.opt = opt
        
        
    def forward(self, attention_feature, text, opt):
        self.lstm.flatten_parameters()
        ctc_output, _ = self.lstm(attention_feature)
        ctc_output = self.fcl(ctc_output)
        ctc_output = F.softmax(ctc_output, dim=-1)
    
#         print('ctc_output : ', ctc_output.shape)
#         if opt.num_gpu >1:
            
#             input_lengths = torch.full(size = (int(self.opt.batch_size/ opt.num_gpu),), fill_value= attention_feature.size(1), dtype=torch.long)
#             output_lengths = torch.randint(low=1, high = attention_feature.size(1),  size = (int(self.opt.batch_size/ opt.num_gpu), ) ,
#                                            dtype=torch.long)
#             ctc_loss = self.ctc_loss(ctc_output.transpose(0,1), text[:, 1:], input_lengths, output_lengths)
        
#         else:
#         input_lengths = torch.full(size = (self.opt.batch_size,), fill_value= attention_feature.size(1), dtype=torch.long)
#         output_lengths = torch.randint(low=1, high = attention_feature.size(1),  size = (self.opt.batch_size, ) ,dtype=torch.long)
#         ctc_loss = self.ctc_loss(ctc_output.transpose(0,1), text[:, 1:], input_lengths.cpu(), output_lengths)
        
#         return ctc_output, ctc_loss
        return ctc_output