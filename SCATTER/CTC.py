import torch
import torch.nn.functional as F

class CTC_decoder(torch.nn.Module):
    def __init__(self, input_size, output_size, num_classes, device ):
        super(CTC_decoder, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, output_size)
        self.fcl = torch.nn.Linear(output_size, num_classes)
        
        
    def forward(self, attention_feature, text):
        self.lstm.flatten_parameters()
        ctc_output, _ = self.lstm(attention_feature)
        ctc_output = self.fcl(ctc_output)
        ctc_output = F.log_softmax(ctc_output, dim=-1)
    
        return ctc_output