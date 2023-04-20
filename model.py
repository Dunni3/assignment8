import torch
import pytorch_lightning as pl
from pathlib import Path
from lang import Language
from torch import optim
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
import transformer as trn
from masked_batch_norm import MaskedBatchNorm1d
import cloudpickle
from torch.distributions import Normal, Categorical

class BindingPredictor(pl.LightningModule):

    def __init__(self,
        seq_map: dict,
        save_interval: float,
        output_dir: Path,
        batches_per_epoch: int,
        max_rec_pred_len: int = 1200,
        n_rec_layers: int = 2,
        n_lig_layers: int = 2,
        n_rec_heads: int = 2,
        n_lig_heads: int = 2,
        warmup_length: float = 0.1,
        dropout: float = 0.33,
        lr: float = 1e-4,
        embedding_dim: int = 512,
        ):
        super().__init__()

        self.lr = lr
        self.warmup_length = warmup_length
        self.batches_per_epoch = batches_per_epoch
        self.max_rec_pred_len = max_rec_pred_len
        self.language = Language()

        self.lig_embedding = nn.Embedding(self.language.smi_lang_size,
                                          embedding_dim=embedding_dim,
                                          padding_idx=self.language.smi_pad_idx)
        self.rec_embedding = nn.Embedding(self.language.rec_lang_size, 
                                          embedding_dim=embedding_dim, 
                                          padding_idx=self.language.rec_pad_idx)
        
        # TODO: this may be faster if you create separate pos encoding objects
        # for lig and rec bc caching of positional encodings
        self.pe = Summer(PositionalEncoding1D(embedding_dim))

        self.lig_encoder = Encoder(embedding_dim=embedding_dim, n_layers=n_lig_layers, n_heads=n_lig_heads, dropout=dropout)
        self.rec_encoder = Encoder(embedding_dim=embedding_dim, n_layers=n_rec_layers, n_heads=n_rec_heads, dropout=dropout)

        self.prediction_mlp = nn.Sequential(
            nn.Linear(embedding_dim*2, embedding_dim*2),
            nn.SiLU(),
            nn.Linear(embedding_dim*2, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.SiLU(),
            nn.Linear(embedding_dim // 4, 1)
        )

        self.loss_fn = nn.MSELoss(reduction='mean')

        self.output_dir = output_dir
        self.save_interval = save_interval
        self.save_marker = 0.0

        self.seq_map = seq_map

    def forward(self, lig_seqs, rec_seqs, lig_masks, rec_masks):

        # expand lig and rec masks for easier broadcasting
        lig_masks = lig_masks[:, :, None]
        rec_masks = rec_masks[:, :, None]
        
        # get embedded sequence tokens
        lig_embedded = self.pe(self.lig_embedding(lig_seqs))*lig_masks
        rec_embedded = self.pe(self.rec_embedding(rec_seqs))*rec_masks

        # pass to encoders
        lig_encoded = self.lig_encoder(lig_embedded, lig_masks) # (batch_size, embedding_dim)
        rec_encoded = self.rec_encoder(rec_embedded, rec_masks) # (batch_size, embedding)dim

        # concat ligand and receptor encodings together and pass through final prediction network
        complex_embedding = torch.concatenate([lig_encoded, rec_encoded], dim=1)

        pred_labels = self.prediction_mlp(complex_embedding).flatten()


        return pred_labels

    def training_step(self, batch_data, batch_idx):
        lig_seqs, rec_seqs, labels, lig_masks, rec_masks = batch_data
        labels = labels.float()

        # truncate ligand and receptor sequences to the length of the longest sequence in the batch
        lig_seq_lens = lig_masks.sum(dim=1)
        rec_seq_lens = rec_masks.sum(dim=1)

        max_lig_len = lig_seq_lens.max()
        max_rec_len  = rec_seq_lens.max()

        if max_lig_len < lig_seqs.shape[1]:
            lig_seqs = lig_seqs[:, :max_lig_len]
            lig_masks = lig_masks[:, :max_lig_len]
        
        if max_rec_len < rec_seqs.shape[1]:
            rec_seqs = rec_seqs[:, :max_rec_len]
            rec_masks = rec_masks[:, :max_rec_len]

        # add extra dim to masks
        # rec_masks = rec_masks.unsqueeze(1)
        # lig_masks = lig_masks.unsqueeze(1)

        epoch_exact = self.current_epoch + batch_idx/self.batches_per_epoch
        self.sched.step_lr(epoch_exact)

        predicted_affinity = self(lig_seqs, rec_seqs, lig_masks, rec_masks)

        # compute loss
        loss = self.loss_fn(predicted_affinity, labels)

        # log loss
        self.log('loss', loss, on_step=True, prog_bar=True)
        self.log('epoch_exact', epoch_exact, on_step=True, prog_bar=True)

        # save model if necessary
        if epoch_exact - self.save_marker > self.save_interval:
            save_file = self.output_dir / f'model_{self.current_epoch}_{batch_idx}.pkl'
            cloudpickle.dump(self, open(save_file, 'wb'))
            self.save_marker = epoch_exact

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.sched = Scheduler(optimizer=optimizer, base_lr=self.lr, warmup_length=self.warmup_length, batches_per_epoch=self.batches_per_epoch)
        return optimizer
    
    def predict(self, smile: str, uniprot: str, device='cuda'):

        rec_str = self.seq_map[uniprot]
        rec_len = len(rec_str)
        smi_seq = self.language.encode_smiles(smile).int()
        rec_seq = self.language.encode_rec(rec_str).int()

        smi_seq = smi_seq.to(device)
        rec_seq = rec_seq.to(device)

        # TODO: reshape sequences
        # TODO: make multiple predictions if len(rec) > self.max_pred_len and take median
        if rec_len > self.max_rec_pred_len:
            max_start_idx = rec_len - self.max_rec_pred_len
            mid_start_idx = max_start_idx/2
            possible_start_idxs = torch.arange(max_start_idx+1)
            
            
            if possible_start_idxs.shape[0] < 32:
                # do all possible windows
                start_idxs = possible_start_idxs
            else:
                normal_dist = Normal(mid_start_idx, mid_start_idx*0.6)
                probs = normal_dist.cdf(possible_start_idxs + 0.5) - normal_dist.cdf(possible_start_idxs - 0.5)
                probs = probs/probs.sum()
                start_idxs = torch.unique(Categorical(probs).sample((48,)))
            
            windows = []
            for start_idx in start_idxs:
                start_idx = int(start_idx)
                windows.append(rec_seq[start_idx:start_idx+self.max_rec_pred_len])

            rec_seq = torch.stack(windows, dim=0)
        else:
            rec_seq = rec_seq[None, :]

        # make lig_seq match the shape of rec_seq by copying out lig_seqs
        smi_seq = smi_seq.repeat(rec_seq.shape[0], 1)

        lig_mask = smi_seq != self.language.smi_pad_idx
        rec_mask = rec_seq != self.language.rec_pad_idx

        predictions = self(smi_seq, rec_seq, lig_mask, rec_mask)
        final_answer = float(torch.median(predictions))
        return final_answer
    

class Scheduler:

    def __init__(self, optimizer, base_lr: float, warmup_length: float, batches_per_epoch: int):

        self.warmup_length = warmup_length
        self.batches_per_epoch = batches_per_epoch
        self.base_lr = base_lr
        self.optimizer = optimizer

    def step_lr(self, epoch_exact: float):
        if epoch_exact <= self.warmup_length:
            self.optimizer.param_groups[0]['lr'] = self.base_lr*epoch_exact/self.warmup_length


class Encoder(nn.Module):

    def __init__(self, embedding_dim: int = 512, n_layers: int = 2, n_heads: int = 3, dropout: float = 0.33):
        super().__init__()

        self.n_layers = n_layers

        self.layers = trn.get_clones(EncoderLayer(embedding_dim, n_heads, dropout), n_layers)
        self.norm = MaskedBatchNorm1d(embedding_dim)

        # components for reduction layer
        self.mean_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), 
            nn.ReLU(), 
            nn.Linear(embedding_dim, embedding_dim), 
            nn.ReLU())
        self.pre_reduct_norm = nn.BatchNorm1d(embedding_dim)
        self.reduction_mha = trn.MultiHeadAttention(heads=4, d_model=embedding_dim, dropout=dropout)
        self.reduction_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), 
            nn.SiLU(), 
            nn.Linear(embedding_dim, embedding_dim), 
            nn.SiLU(),
            nn.BatchNorm1d(embedding_dim))

    def forward(self, x, mask):
        # x has shape (batch_size, seq_len, embedding_dim)
        # mask has shape (batch_size, seq_len, 1)

        for i in range(self.n_layers):
            x = self.layers[i](x, mask)
        x = self.norm(x, mask)

        mean_val = x.mean(dim=1, keepdim=True)
        mean_val = self.pre_reduct_norm(self.mean_mlp(mean_val).transpose(1, 2)).transpose(1, 2)
        x = self.reduction_mha(mean_val, x, x, mask[:, :, 0], reduction=True).squeeze(1)
        x = self.reduction_mlp(x)

        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = MaskedBatchNorm1d(d_model)
        self.norm_2 = MaskedBatchNorm1d(d_model)
        # self.norm_1 = trn.Norm(d_model)
        # self.norm_2 = trn.Norm(d_model)
        self.attn = trn.MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = trn.FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # x has shape (batch_size, seq_len, embedding_dim)
        # mask has shape (batch_size, seq_len, 1)
        x2 = self.norm_1(x, mask)
        x = x + self.dropout_1(self.attn(x2,x2,x2, mask[:, :, 0]))
        x2 = self.norm_2(x, mask)
        x = x + self.dropout_2(self.ff(x2)*mask)
        return x
    
