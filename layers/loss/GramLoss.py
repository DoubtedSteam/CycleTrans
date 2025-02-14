import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
def Gram_matrix(input):
    b, c, h, w = input.size()
    F = input.view(b, c, h*w)
    G = torch.bmm(F, F.transpose(1, 2))
    G.div_(h*w)
    return G


class GramLoss(nn.Module):
    def forward(self,x,sub):
        x_v = x[sub]
        x_r = x[~sub]
        gram_v = Gram_matrix(x_v)
        gram_r = Gram_matrix(x_r)
        Loss = nn.MSELoss()(x_v,x_r)
        # for i ,pred_v in enumerate(gram_v): 
        #     # pred_v = pred_v.cpu().detach().numpy()
        #     pred_v = np.uint8(pred_v.cpu().detach()*255)
        #     row, col = np.diag_indices_from(pred_v)
        #     pred_v[row,col] = 0
        #     plt.imshow(pred_v, cmap=plt.cm.jet)
        #     plt.savefig('gram/gram_v_{}.png'.format(i))
        return Loss
