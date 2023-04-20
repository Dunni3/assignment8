import torch

class Language:

    def __init__(self):

        self.smi_char_to_idx = {'H': 0, ')': 1, '4': 2, '@': 3, 'M': 4, ']': 5, 'K': 6, '=': 7, 
                                '\\': 8, '+': 9, 'P': 10, 'O': 11, '#': 12, 'R': 13, '3': 14, 'E': 15, 
                                '/': 16, '9': 17, 'I': 18, '1': 19, '.': 20, 'F': 21, 'B': 22, '6': 23, 
                                'C': 24, '-': 25, '[': 26, 'Z': 27, '7': 28, 'S': 29, '(': 30, '8': 31, 
                                'A': 32, '5': 33, 'T': 34, 'L': 35, 'N': 36, 'G': 37, '2': 38, 'pad': 39}
        
        self.rec_char_to_idx = {'H': 0, 'Q': 1, 'M': 2, 'K': 3, 'W': 4, 'P': 5, 'R': 6, 'E': 7, 'X': 8, 
                                'I': 9, 'F': 10, 'B': 11, 'V': 12, 'C': 13, 'U': 14, 'Z': 15, 'S': 16, 'T': 17, 
                                'A': 18, 'D': 19, 'L': 20, 'N': 21, 'Y': 22, 'G': 23, 'pad': 24}
        
    @property
    def smi_pad_idx(self):
        return self.smi_char_to_idx['pad']
    
    @property
    def rec_pad_idx(self):
        return self.rec_char_to_idx['pad']
    
    @property
    def rec_lang_size(self):
        return len(self.rec_char_to_idx)
    
    @property
    def smi_lang_size(self):
        return len(self.smi_char_to_idx)
        
    def encode_smiles(self, smi_str: str):
        smi_idxs = [ self.smi_char_to_idx[char] for char in smi_str.upper() ]
        return torch.tensor(smi_idxs, dtype=torch.uint8)
    
    def encode_rec(self, rec_seq: str):
        rec_idxs = [ self.rec_char_to_idx[char] for char in rec_seq.upper() ]
        return torch.tensor(rec_idxs, dtype=torch.uint8)