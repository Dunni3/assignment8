import argparse
import gzip
import pandas as pd

from process_training_data import get_uniprot_seq_map

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--uniprot_seq_map', type=str, default='data/kinase_seqs_expanded.txt')
    p.add_argument('--test_data', default='data/test1_labels.txt')
    p.add_argument('--output_file', type=str, default=None)

    # p.add_argument('--max_rec_len', type=int, default=2600)
    # p.add_argument('--max_lig_len', type=int, default=100)

    args = p.parse_args()
    return args

def parse_test_data(test_data_file):

    df = pd.read_csv(test_data_file, delimiter=' ')

    if 'Unnamed' in df.columns.tolist()[-1]:
        df = df.drop(columns=df.columns.tolist()[-1])

    return df

if __name__ == "__main__":
    args = parse_args()

    # get chembl data as dataframe
    test_data = parse_test_data(args.test_data)
    
    # get mapping from uniprot id -> AA sequence
    uniprot_seq_map = get_uniprot_seq_map(args.uniprot_seq_map)

    # apply uniprot -> seq map to data
    test_data['seq'] = test_data['UniProt'].map(uniprot_seq_map)

    # get all data for which a target sequence was found
    mapped_data = test_data[ ~test_data['seq'].isna() ]
    

    seqs_found = test_data.shape[0] - test_data['seq'].isna().sum()

    # print(f'{uniprot_ids_found*100/test_data.shape[0]:.2f}% of chembl targets mapped to uniprot ids')
    # print(f'{seqs_found*100/uniprot_ids_found:.2f}% of uniprot ids mapped to sequences')
    # print(f'{seqs_found} sequences found')
    assert seqs_found == test_data.shape[0]

    if args.output_file is not None:
        if mapped_data.shape[1] == 3:
            mapped_data = mapped_data[['SMILES', 'seq']]
            mapped_data.columns = ['smiles', 'seq']
        elif mapped_data.shape[1] == 4:
            mapped_data = mapped_data[['SMILES', 'seq', 'pKd']]
            mapped_data.columns = ['smiles', 'seq', 'label']
        else:
            print(mapped_data.columns)
            raise ValueError
        mapped_data.to_pickle(args.output_file)