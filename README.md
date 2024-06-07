# GADY: Unsupervised Anomaly Detection on Dynamic Graphs

This is a demo for "GADY: Unsupervised Anomaly Detection on Dynamic Graphs"

## Data
You can download the datasets from:
* UCI: http://konect.cc/networks/opsahl-ucsocial
* Email-DNC: http://networkrepository.com/email-dnc
* Bitcoin-alpha: https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html

Once you do so, place the files in the 'data' folder and run, e.g:
```{bash}
python prepare_data.py --dataset uci
```

## Preprocessing

To run GADY on any dataset, we first precompute the positional features. We'll use uci as a running example.
We start off doing so for the training data:
```{bash}
python preproc_new.py --data uci --gpu 0 --r_dim 4 --data_split train --anomaly_per 0.1
python preproc_new.py --data uci --gpu 0 --r_dim 4 --data_split train --anomaly_per 0.05
python preproc_new.py --data uci --gpu 0 --r_dim 4 --data_split train --anomaly_per 0.01
```
The flag 'r-dim' sets the dimension of positional features. 

Then, we do the same for the test splits:
```{bash}
python preproc_new.py --data uci --gpu 0 --r_dim 4 --data_split test --anomaly_per 0.1
python preproc_new.py --data uci --gpu 0 --r_dim 4 --data_split test --anomaly_per 0.05
python preproc_new.py --data uci --gpu 0 --r_dim 4 --data_split test --anomaly_per 0.01
```

## Running GADY
With the precomputed positional features at hand, we run GADY using the following commands.

For UCI:
```{bash}
python train.py --data uci --mode 0 --gpu 0 --anomaly_per 0.1 --alpha 0.1 --betaa 10 --gamma 0.1 --n_layer 2 --use_memory --beta 0.00001 --n_epoch 50 --patience 5 --n_runs 6 --n_degree 10 --memory_dim 172 
```

For btc_otc:
```{bash}
python train.py --data btc_otc --mode 0 --gpu 0 --anomaly_per 0.1 --alpha 0.1 --betaa 10 --gamma 0.1 --n_layer 2 --use_memory --beta 0.00001 --n_epoch 50 --patience 5 --n_runs 6 --n_degree 10 --memory_dim 172
```
For email_dnc:
```{bash}
python train.py --data email_dnc --mode 0 --gpu 0 --anomaly_per 0.1 --alpha 0.1 --betaa 10 --gamma 0.1 --n_layer 2 --use_memory --beta 0.00001 --n_epoch 50 --patience 5 --n_runs 6 --n_degree 10 --memory_dim 172
```