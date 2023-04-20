import torch
from lang import Language
import pandas as pd
from pathlib import Path

from torch.distributions import Normal, Categorical

class Dataset(torch.utils.data.Dataset):

    def __init__(self, pickle_file_path: str, max_lig_len=100, max_pred_rec_len=1200, max_save_rec_len=2600):
        super().__init__()

        self.pickle_file_path = Path(pickle_file_path)

        self.max_lig_len = max_lig_len
        self.max_pred_rec_len = max_pred_rec_len
        self.max_save_rec_len = max_save_rec_len
        self.language = Language()

        self.process()

    def process(self):
        # get name of pre-processed dataset
        preprocessed_file = self.pickle_file_path.parent / f'{self.pickle_file_path.stem}_preprocessed.pt'

        if preprocessed_file.exists():
            data_dict = torch.load(str(preprocessed_file))
            self.ligs = data_dict['ligs']
            self.recs = data_dict['recs']
            self.labels = data_dict['labels']
            self.lig_masks = data_dict['lig_masks']
            self.rec_masks = data_dict['rec_masks']
            return
        
        # if the preprocessed file does not exist, we must create it
        unprocessed_data = pd.read_pickle(self.pickle_file_path).reset_index(drop=True)

        # get lengths of receptors
        rec_lengths = unprocessed_data['seq'].str.len()
        data_max_rec_len = rec_lengths.max()
        save_len = min(data_max_rec_len, self.max_save_rec_len)

        # initialize data tensors
        nrows = unprocessed_data.shape[0]
        encoded_ligs = torch.full((nrows, self.max_lig_len), fill_value=self.language.smi_pad_idx, dtype=torch.uint8)
        encoded_recs = torch.full((nrows, save_len), fill_value=self.language.rec_pad_idx, dtype=torch.uint8)

        # fill data tensors with encoded sequences
        for i, row in unprocessed_data.iterrows():
            encoded_lig = self.language.encode_smiles(row['smiles'])
            encoded_rec = self.language.encode_rec(row['seq'])

            encoded_ligs[i, :encoded_lig.shape[0]] = encoded_lig
            encoded_recs[i, :encoded_rec.shape[0]] = encoded_rec

            if i % 100000 == 0:
                print(f'{i + 1} data examples processed')

        lig_masks = encoded_ligs != self.language.smi_pad_idx
        rec_masks = encoded_recs != self.language.rec_pad_idx

        # convert labels to torch tensor
        labels = torch.tensor(unprocessed_data['label'])

        # save processed data
        torch.save({
            'ligs': encoded_ligs,
            'recs': encoded_recs,
            'labels': labels,
            'lig_masks': lig_masks,
            'rec_masks': rec_masks
        }, str(preprocessed_file))

        self.ligs = encoded_ligs
        self.recs = encoded_recs
        self.labels = labels
        self.lig_masks = lig_masks
        self.rec_masks = rec_masks

    def __len__(self):
        return self.ligs.shape[0]
    
    def __getitem__(self, idx):

        rec_seq = self.recs[idx]
        rec_mask = self.rec_masks[idx]
        rec_len = rec_mask.sum()

        # if the amino acid sequence is longer than the maximum prediction length
        if rec_len > self.max_pred_rec_len:
            # randomly select a window
            max_start_idx = rec_len - self.max_pred_rec_len
            mid_start_idx = max_start_idx/2
            possible_start_idxs = torch.arange(max_start_idx+1)
            normal_dist = Normal(mid_start_idx, mid_start_idx*0.6)
            probs = normal_dist.cdf(possible_start_idxs + 0.5) - normal_dist.cdf(possible_start_idxs - 0.5)
            probs = probs/probs.sum()
            start_idx = Categorical(probs).sample((1,))[0] # start index for window of length self.max_pred_rec_len
            rec_seq_window = rec_seq[start_idx:start_idx+self.max_pred_rec_len] # select our window of sequence to predict on
            rec_seq = torch.ones_like(rec_seq)*self.language.rec_pad_idx
            rec_seq[:rec_seq_window.shape[0]] = rec_seq_window
            rec_mask = rec_seq != self.language.rec_pad_idx


        return self.ligs[idx].int(), rec_seq.int(), self.labels[idx], self.lig_masks[idx], rec_mask