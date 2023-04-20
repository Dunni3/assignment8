import argparse
import gzip
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--uniprot_seq_map', type=str, default='data/kinase_seqs_expanded.txt')
    p.add_argument('--chembl_uniprot_map', default='data/chembl_uniprot_mapping.txt')
    p.add_argument('--chembl_data', default='data/chembl_activities.txt.gz')
    p.add_argument('--save_unmapped_uniprot', action='store_true')
    p.add_argument('--output_file', type=str, default=None)


    p.add_argument('--max_rec_len', type=int, default=2600)
    p.add_argument('--max_lig_len', type=int, default=100)
    args = p.parse_args()
    return args

def parse_chembl_data(chembl_data_file):

    with gzip.open(chembl_data_file, 'rt') as f:
        df = pd.read_csv(f, delimiter=' ')

    return df

def get_chembl_uniprot_map(map_file):

    chembl_uniprot_map = {}

    with open(map_file, 'r') as f:
        next(f)
        for line in f:
            split_line = line.strip().split('\t')

            if split_line[-1] != "SINGLE PROTEIN":
                continue

            uniprot_id, chembl_id = split_line[:2]

            chembl_uniprot_map[chembl_id] = uniprot_id

    return chembl_uniprot_map

def get_uniprot_seq_map(map_file):

    uniprot_seq_map = {}

    with open(map_file, 'r') as f:
        for line in f:
            uniprot_id, seq = line.strip().split(' ')[:2]
            uniprot_seq_map[uniprot_id] = seq

    return uniprot_seq_map

def filter_by_length(df, max_rec_len, max_lig_len):
    rec_lens = df['seq'].str.len()
    lig_lens = df['smiles'].str.len()

    mask = (rec_lens <= max_rec_len) & (lig_lens <= max_lig_len)
    df = df[mask]
    return df

if __name__ == "__main__":
    args = parse_args()

    # get chembl data as dataframe
    chembl_data = parse_chembl_data(args.chembl_data)

    # get mapping from chembl targets -> uniprot id
    chembl_uniprot_map = get_chembl_uniprot_map(args.chembl_uniprot_map)
    
    # get mapping from uniprot id -> AA sequence
    uniprot_seq_map = get_uniprot_seq_map(args.uniprot_seq_map)
    
    # apply chembl -> uniprot map to data
    chembl_data['uniprot_id'] = chembl_data['target'].map(chembl_uniprot_map)

    # apply uniprot -> seq map to data
    chembl_data['seq'] = chembl_data['uniprot_id'].map(uniprot_seq_map)

    # get all data for which a target sequence was found
    mapped_data = chembl_data[ ~chembl_data['seq'].isna() ]
    

    uniprot_ids_found = chembl_data.shape[0] - chembl_data['uniprot_id'].isna().sum()
    seqs_found = chembl_data.shape[0] - chembl_data['seq'].isna().sum()

    print(f'{uniprot_ids_found*100/chembl_data.shape[0]:.2f}% of chembl targets mapped to uniprot ids')
    print(f'{seqs_found*100/uniprot_ids_found:.2f}% of uniprot ids mapped to sequences')
    print(f'{seqs_found} sequences found')

    if args.save_unmapped_uniprot:
        mask = ( ~chembl_data['uniprot_id'].isna() ) & ( chembl_data['seq'].isna() )
        unmapped_uniprot_ids = chembl_data[ mask ]['uniprot_id'].unique().tolist()
        with open('data/unmapped_uniprot_ids.txt', 'w') as f:
            f.write('\n'.join(unmapped_uniprot_ids))

    if args.output_file is not None:
        mapped_data = mapped_data[['smiles', 'seq', 'pchembl']]
        mapped_data.columns = ['smiles', 'seq', 'label']

        mapped_data = filter_by_length(mapped_data, max_rec_len=args.max_rec_len, max_lig_len=args.max_lig_len)
        print(f'{mapped_data.shape[0]} datapoints saved')

        mapped_data.to_pickle(args.output_file)