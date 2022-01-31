import torch
import torch.nn as nn

class CustomBCELoss(nn.Module):
    
    def __init__(self):
        super(CustomBCELoss, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, y_hat, y):
        
        y_hat = y_hat.view(-1)
        y = y.view(-1)

        y_hat = y_hat[y > -0.5]
        y = y[y > -0.5]  
        
        return self.loss(y_hat, y) 


class CustomBCEWithLogitsLoss(nn.Module):
    
    def __init__(self):
        super(CustomBCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        
    def forward(self, y_hat, y):
        
        y_hat = y_hat.view(-1)
        y = y.view(-1)

        y_hat = y_hat[y > -0.5]
        y = y[y > -0.5]  
        
        return self.loss(y_hat, y)
  

   