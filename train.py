import argparse
import uuid
from pathlib import Path

# import cloudpickle # cloudpickle provides the most robust saving of objects
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
import sys

from dataset import Dataset
from lang import Language
from model import BindingPredictor
from process_training_data import get_uniprot_seq_map


def parse_args():
    p = argparse.ArgumentParser(description='Random predictions.')
    # p.add_argument('--map',type=str, help='map from uniprot to chembl target of interest',required=True)
    # p.add_argument('--affinities',type=str, help='ChEMBL affinities file',required=True)
    # p.add_argument('--out',type=str, default='model.pkl', help="Output model")
    p.add_argument('--seq_map', type=str, default='data/kinase_seqs_expanded.txt')

    p.add_argument('--train_data', type=str, default='clean_data/train.pkl')
    p.add_argument('--test_data', nargs='+', default=['clean_data/test_1.pkl', 'clean_data/test_1.pkl'])
    p.add_argument('--dev_run', action='store_true')

    p.add_argument('--n_rec_layers', type=int, default=2)
    p.add_argument('--n_lig_layers', type=int, default=2)
    p.add_argument('--n_rec_heads', type=int, default=2)
    p.add_argument('--n_lig_heads', type=int, default=2)
    p.add_argument('--embedding_dim', type=int, default=512)

    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warmup_length', type=float, default=0.027)
    p.add_argument('--max_rec_len', type=int, default=700)

    p.add_argument('--save_interval', type=float, default=0.1)
    p.add_argument('--log_interval', default=1e-3, type=float)
    p.add_argument('--output_dir', type=str, default='training_runs/')

    p.add_argument('--accumulate_grad_batches', type=int, default=1)

    p.add_argument('--seed', type=int, default=42)

    args = p.parse_args()
    return args

if __name__ == "__main__":
    # model = Model(args.map, args.affinities)
    # cloudpickle.dump(model,open(args.out,'wb'))
    args = parse_args()
    seed_everything(args.seed)

    # construct uniprot -> seq map
    uniprot_seq_map = get_uniprot_seq_map(args.seq_map)

    train_dataset = Dataset(args.train_data, max_pred_rec_len=args.max_rec_len)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=3)

    batches_per_epoch = len(train_dataloader)
    steps_per_log = int( batches_per_epoch*args.log_interval )
    if steps_per_log == 0:
        steps_per_log = 1

    lr_monitor = LearningRateMonitor(logging_interval='step')

    if not args.dev_run:
        wandb_logger = WandbLogger(project='assignment8', group='sweep')
    else:
        wandb_logger = WandbLogger(project='assignment8', name='test', mode='disabled')
    
    # determine output dir
    results_dir = Path(args.output_dir)
    results_dir.mkdir(exist_ok=True)
    random_id = str(uuid.uuid1())[:4]
    if wandb.run is not None:
        output_dir = results_dir / f'{wandb.run.name}-{random_id}'
    else:
        output_dir = results_dir / f'{random_id}'

    output_dir.mkdir()

    model = BindingPredictor(seq_map=uniprot_seq_map,
                             save_interval=args.save_interval,
                             output_dir=output_dir,
                             batches_per_epoch=batches_per_epoch,
                             n_rec_layers=args.n_rec_layers,
                             n_lig_layers=args.n_lig_layers,
                             n_rec_heads=args.n_rec_heads,
                             n_lig_heads=args.n_lig_heads,
                             warmup_length=args.warmup_length,
                             lr=args.lr,
                             embedding_dim=args.embedding_dim,
                             max_rec_pred_len=args.max_rec_len)
    

    # model.eval()
    # with torch.no_grad():
    #     val = model.predict("C[C@@H](OC1=C(N)N=CC(C2=C(C#N)N(C)N=C2C3)=C1)C4=CC(F)=CC=C4C(N3C)=O", "Q07912")
    #     print('meep')

    if not args.dev_run: 
        trainer = pl.Trainer(accelerator='gpu', callbacks=lr_monitor, max_epochs=10, devices=1, log_every_n_steps=steps_per_log, logger=wandb_logger)
    else:
        trainer = pl.Trainer(accelerator='gpu', callbacks=lr_monitor, max_steps=20, devices=1, log_every_n_steps=steps_per_log, logger=wandb_logger, accumulate_grad_batches=args.accumulate_grad_batches)

    trainer.fit(model, train_dataloader)