from pathlib import Path
import pandas as pd

data_dir = Path('clean_data')

dfs = [ pd.read_pickle(fp) for fp in data_dir.iterdir()  ]

smi_set, rec_set = set(), set()

for df in dfs:
    all_smiles = ''.join(df['smiles'].unique().tolist()).upper()
    all_seqs = ''.join(df['seq'].unique().tolist()).upper()
    smi_set.update(all_smiles)
    rec_set.update(all_seqs)

# print(smi_set)
# print(rec_set)

smi_char_to_idx = {}
for i, char in enumerate(smi_set):
    smi_char_to_idx[char] = i
print(smi_char_to_idx)

rec_char_to_idx = {}
for i, char in enumerate(rec_set):
    rec_char_to_idx[char] = i
print(rec_char_to_idx)