import torch

class CTC_decoder(torch.nn.Module):
    def __init__(self, input_size, output_size, num_classes, opt ):
        super(CTC_decoder, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, output_size)
        self.fcl = torch.nn.Linear(output_size, num_classes)
        self.ctc_loss = torch.nn.CTCLoss(blank= 0, reduction='mean')
        self.opt = opt
        
        
    def forward(self, attention_feature, text):
        ctc_output, _ = self.lstm(attention_feature)
        ctc_output = self.fcl(ctc_output)
        ctc_output = torch.softmax(ctc_output, dim=-1)
#         ctc_output = torch.argmax(ctc_output, dim=-1, keep_dim=True)
#         print(ctc_output.shape)
        input_lengths = torch.full(size = (self.opt.batch_size,), fill_value= attention_feature.size(1), dtype=torch.long)
        output_lengths = torch.randint(low=1, high = attention_feature.size(1),  size = (self.opt.batch_size, ) , dtype=torch.long)
        ctc_loss = self.ctc_loss(ctc_output.transpose(0,1), text[:, 1:], input_lengths, output_lengths)
        
        return ctc_output, ctc_loss