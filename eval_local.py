#!/usr/bin/env python3
import cloudpickle, sys
import fsspec
import torch

DEVICE = 'cuda'

#fetch and load model from Google Cloud
# model = cloudpickle.load(fsspec.open_files("gs://mscbio2066-data/model.pkl","rb")[0].open())
model = cloudpickle.load(open('training_runs/comfy-thunder-66-e6d5/model_0_32.pkl', 'rb'))
model.to(DEVICE)


infile = open(sys.argv[1])
out = open(sys.argv[2],'wt')

out.write(infile.readline()) # header

model.eval()
with torch.no_grad():
    for line in infile:
        smile,uniprot = line.strip().split()
        val = model.predict(smile,uniprot)
        out.write(f'{smile} {uniprot} {val:.4f}\n')