import math
import logging
import time
import sys
import argparse
import torch
import os
import numpy as np
from pathlib import Path
from modules.GAN import Generator
from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder, get_data_settings, GenFGANLoss, DiscFGANLoss
from utils.data_processing import get_data, compute_time_statistics

torch.autograd.set_detect_anomaly(True)
cpu_num = 10 
os.environ["OMP_NUM_THREADS"] = str(cpu_num)  
os.environ["MKL_NUM_THREADS"] = str(cpu_num) 
torch.set_num_threads(cpu_num )


### Argument and global variables
parser = argparse.ArgumentParser('GADY, Unsupervised learning')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='uci')
parser.add_argument('--seed', type=int, default=142, help='seed')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_epoch', type=int, default=1, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=1, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=100, help='Dimensions of the memory for '
                                                                'each user') # was 172, 100 for uci
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')

parser.add_argument('--mode', type=int, default=0,
                    help='running mode for the GADY, set it to 1 to run ablation study')

parser.add_argument('--beta', type=float, default=0.1,
                    help='Initial value for the beta parameter in GADY')
parser.add_argument('--r_dim', type=int, default=4, help='dim of positional features')

parser.add_argument('--lr_G', type=float, default=0.000001, help='Learning rate of generator')
parser.add_argument('--lr_D', type=float, default=0.000001, help='Learning rate of discriminator')
parser.add_argument('--anomaly_per', type=float, default=0.1, help='the anomaly rate of the dataset')



try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(args.data+'_'+str(args.anomaly_per)+'_'+str(time.time()))))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)
alpha = args.alpha
betaa = args.betaa
gamma = args.gamma


### Extract data for training, validation and testing
node_features, edge_features, full_data, train_data, test_data,  = get_data(args.data,
                              different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features,anomaly_per = args.anomaly_per)

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
if args.mode == 1:   # ablation study for the GADY
  train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)

# Set device
device_string = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)
performance_best_auc = 0
performance_best_ap = 0
train_generator = Generator(n_neighbors=args.n_degree,batch_size=args.bs,device = device)


for i in range(args.n_runs):
  results_path = f"results/GADY-{args.data}_{i}.pkl" if args.prefix == '' else f"results/{args.prefix}_{i}.pkl"
  Path("results/").mkdir(parents=True, exist_ok=True)

  # Initialize Model
  discriminator = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=args.n_layer, use_memory=args.use_memory,
            message_dimension=args.message_dim, memory_dimension=args.memory_dim,
            memory_update_at_start=not args.memory_update_at_end,
            message_function=args.message_function,
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=args.n_degree,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            beta=args.beta,
            r_dim=args.r_dim,
            Generator = train_generator)
  criterion = torch.nn.BCELoss()
  d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr_D)
  g_optimizer = torch.optim.Adam(discriminator.Generator.parameters(), lr=args.lr_G)
  discriminator = discriminator.to(device)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / args.bs)

  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)


  test_aps = []
  test_aucs = []
  epoch_times = []
  total_epoch_times = []
  train_losses = []
  generator_losses = []
  discriminator_losses = []
  
  G_loss = GenFGANLoss()
  D_loss = DiscFGANLoss()


  early_stopper = EarlyStopMonitor(max_round=args.patience)

  next_V, next_R = [], []

  test_V, test_R = [], []

  for epoch in range(args.n_epoch):
    discriminator.reset_VR()
    start_epoch = time.time()

    ### Training
    # Reinitialize memory of the model at the start of each epoch
    if args.use_memory:
      discriminator.memory.__init_memory__()

    # Train using only training graph
    discriminator.set_neighbor_finder(train_ngh_finder)
    m_loss = []
    m_loss2 = []
    
    logger.info('start {} epoch'.format(epoch))
    t_epoch_begin = time.time()
    for k in range(0, num_batch):
      d_loss = 0
      g_loss = 0
      d_optimizer.zero_grad()

      # Custom loop to allow to perform
      # backpropagation only every a certain number of batches
      batch_idx = k

      if batch_idx >= num_batch:
        continue

      start_idx = batch_idx * args.bs
      end_idx = min(num_instance, start_idx + args.bs)
      sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                          train_data.destinations[start_idx:end_idx]
      edges = np.hstack((sources_batch.reshape(-1,1), destinations_batch.reshape(-1,1)))
      edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
      timestamps_batch = train_data.timestamps[start_idx:end_idx]
      batch_data = np.hstack((edges,timestamps_batch.reshape(-1,1)))
      size = len(sources_batch)
      sources_batch = torch.tensor(sources_batch)
      destinations_batch = torch.tensor(destinations_batch)
      timestamps_batch = torch.tensor(timestamps_batch)

      size = len(sources_batch)
      # _, negatives_batch = train_rand_sampler.sample(size)

      with torch.no_grad():
        pos_label = torch.zeros(size, dtype=torch.float, device=device)
        neg_label = torch.ones(size, dtype=torch.float, device=device)

      discriminator = discriminator.train()

      partition_size, _ = get_data_settings(args.data)

      idx = k
      if (k % partition_size == 0) and args.data: # savepoint
        prt = k // partition_size
        next_V, next_R = torch.load( 'pos_features/' + args.data +'_nextVR_part_' + str(prt) + '_bs_' + str(args.bs) +
                                      '_rdim_' +str(args.r_dim)+ str(args.anomaly_per))
        for c in range(len(next_V)):
          next_V[c] = next_V[c].to(device)
          next_R[c] = next_R[c].to(device)

      if args.mode == 0:
        # neg_samples = train_generator(batch_data)
        # print("hhhhhh")
        
        discriminator.train()
        discriminator.Generator.eval()
        pos_prob = discriminator.compute_edge_probabilities(sources_batch, destinations_batch, timestamps_batch, edge_idxs_batch, args.n_degree,
                                        update_memory=True,
                                  next_V=discriminator.V + next_V[idx%partition_size].to_dense(),
                                  next_R=discriminator.R + next_R[idx%partition_size].to_dense()) 
        neg_prob2,_= discriminator.compute_neg_edge_probabilities(sources_batch, destinations_batch,
                                            timestamps_batch, edge_idxs_batch, args.n_degree,update_memory=False,
                                            next_V=discriminator.V + next_V[idx%partition_size].to_dense(),
                                            next_R=discriminator.R + next_R[idx%partition_size].to_dense(),
                                            nograd = True)
        d_loss = D_loss(neg_prob2.squeeze(),pos_prob.squeeze())
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        discriminator.eval()        
        discriminator.Generator.train()
        neg_prob, neg_samples= discriminator.compute_neg_edge_probabilities(sources_batch, destinations_batch, timestamps_batch,
                                  edge_idxs_batch, args.n_degree,update_memory=False,
                                  next_V=discriminator.V + next_V[idx%partition_size].to_dense(),
                                  next_R=discriminator.R + next_R[idx%partition_size].to_dense())
        g_loss = G_loss(neg_prob.squeeze(), neg_samples)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
      if args.mode == 1:
        _, negatives_batch = train_rand_sampler.sample(edges)
        pos_prob = discriminator.compute_edge_probabilities(sources_batch, destinations_batch, timestamps_batch,edge_idxs_batch, args.n_degree,
                                                  update_memory=True,
                                  next_V=discriminator.V + next_V[idx%partition_size].to_dense(),
                                  next_R=discriminator.R + next_R[idx%partition_size].to_dense())
        d_real_loss = criterion(pos_prob.squeeze(), pos_label)
        
        neg_prob = discriminator.compute_edge_probabilities(sources_batch, negatives_batch,timestamps_batch,
                                            edge_idxs_batch, args.n_degree,update_memory=False,
                                            next_V=discriminator.V + next_V[idx%partition_size].to_dense(),
                                            next_R=discriminator.R + next_R[idx%partition_size].to_dense())
        d_fake_loss = criterion(neg_prob.squeeze(), neg_label)
        d_loss = d_fake_loss + d_real_loss
        d_loss.backward()
        d_optimizer.step()

      m_loss.append(d_loss.item())
      if args.mode == 0:
        m_loss2.append(g_loss.item())
      

    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    train_V, train_R = discriminator.V, discriminator.R

    test_V, test_R = torch.load('pos_features/' + args.data + '_VR_test_bs_' + str(args.bs) +
                                                    '_rdim_' +str(args.r_dim) +str(args.anomaly_per))
    discriminator.set_neighbor_finder(full_ngh_finder)

    if args.use_memory:
      train_memory_backup = discriminator.memory.backup_memory()

    test_ap, test_auc, _, _ = eval_edge_prediction(model=discriminator, negative_edge_sampler=test_rand_sampler, data=test_data,
                                                 n_neighbors=args.n_degree, vs=test_V, rs=test_R)
    if args.use_memory:
      val_memory_backup = discriminator.memory.backup_memory()
      discriminator.memory.restore_memory(train_memory_backup)
      
    test_aucs.append(test_auc)
    test_aps.append(test_ap)
    if args.mode == 0:
      generator_losses.append(np.mean(m_loss2))
    train_losses.append(np.mean(m_loss))
    if test_ap > performance_best_ap:
      performance_best_ap = test_ap
    if test_auc > performance_best_auc:
      performance_best_auc = test_auc

    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)
    logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    if args.mode == 0:
      logger.info('Epoch mean G_loss: {}'.format(np.mean(m_loss2)))
    logger.info(
      'test auc: {}'.format(test_auc))
    logger.info(
      'test ap: {}'.format(test_ap))
    logger.info(
      'best auc: {}'.format(performance_best_auc))
    logger.info(
      'best ap: {}'.format(performance_best_ap))

    # Early stopping
    if early_stopper.early_stop_check(test_ap):
      logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
      break

  test_R, test_V, next_V, next_R = [], [], [], []

  if args.use_memory:
    discriminator.memory.restore_memory(train_memory_backup)
