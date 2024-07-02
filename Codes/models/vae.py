import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class Decoder(nn.Module):
    def __init__(self, dim_in, dim_hidden, n_layers, n_heads=4, max_seq_len=600) -> None:
        super().__init__()

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_in, nhead=3, dim_feedforward=dim_hidden, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.out = nn.Sequential(
            nn.Linear(dim_in, dim_hidden)
        )

    def forward(self, xa_in: torch.Tensor):
        output = self.out(xa_in)            # [:, :-1, :]
        return output.contiguous()


    def loss_function(self, recons: torch.Tensor, input: torch.Tensor):
        # kld_weight = 0.0005
        recons_loss = F.l1_loss(recons, input)


        # kld_loss = nn.KLDivLoss(reduction='batchmean')(recons, input)

        loss = recons_loss #+ kld_weight * kld_loss
        return loss
