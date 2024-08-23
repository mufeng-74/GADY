import numpy as np
import random
import pandas as pd


class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
    self.sources = sources
    self.destinations = destinations
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.labels = labels
    self.n_interactions = len(sources)
    self.unique_nodes = set(sources) | set(destinations)
    self.n_unique_nodes = len(self.unique_nodes)


def get_data_node_classification(dataset_name, use_validation=False):
  ### Load data and train val test split
  graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
  edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
  node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name))

  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  random.seed(2020)

  train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
  test_mask = timestamps > test_time
  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if use_validation else test_mask

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])

  return full_data, node_features, edge_features, train_data, val_data, test_data


def get_data(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False, anomaly_per = None, train_per = 0.7):
  ### Load data and train val test split
  if anomaly_per == None:
    data_train = np.load('./data/'+dataset_name+'_semi_'+'train.npy')
    data_test = np.load('./data/'+dataset_name+'_semi_'+'test.npy')
  else:
    data_train = np.load('./data/'+dataset_name+str(anomaly_per)+'train.npy')
    data_test = np.load('./data/'+dataset_name+str(anomaly_per)+'test.npy')
  # train_feat = np.zeros((data_train.shape[0], 172))
  data_full = np.vstack((data_train,data_test))
  train_num = np.size(data_train,0)
  # graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
  # edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
  edge_features = np.zeros((data_full.shape[0], 172))
  node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name)) 
  
  if randomize_features:
    node_features = np.random.rand(node_features.shape[0], node_features.shape[1])


  data_train = data_train.astype(int)
  data_test = data_test.astype(int)
  
  sources = data_full[:,0].astype(int)
  destinations = data_full[:,1].astype(int)
  edge_idxs = data_full[:,4].astype(int)
  timestamps = data_full[:,2].astype(int)
  if anomaly_per == None:
    labels = data_full[:,3].astype(int)
  else:
    labels = (np.ones_like(data_full[:,3])-data_full[:,3]).astype(int)
  
  full_data =  Data(sources, destinations, timestamps, edge_idxs, labels)

  random.seed(2020)
  # val_time, test_time = list(np.quantile(timestamps, [0.70, 0.85]))
  if anomaly_per == None:
    train_data = Data(data_train[:,0], data_train[:,1], data_train[:,2] ,
                      data_train[:,4], data_train[:,3])

    test_data = Data(data_test[:,0], data_test[:,1], data_test[:,2] ,
                      data_test[:,4], data_test[:,3])
  else:
  
    train_data = Data(data_train[:,0], data_train[:,1], data_train[:,2] ,
                      data_train[:,4], np.ones_like(data_train[:,3])-data_train[:,3])

    test_data = Data(data_test[:,0], data_test[:,1], data_test[:,2] ,
                      data_test[:,4], np.ones_like(data_test[:,3])-data_test[:,3])


  print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                      full_data.n_unique_nodes))
  print("The training dataset has {} interactions, involving {} different nodes".format(
    train_data.n_interactions, train_data.n_unique_nodes))
  # print("The validation dataset has {} interactions, involving {} different nodes".format(
  #   val_data.n_interactions, val_data.n_unique_nodes))
  print("The test dataset has {} interactions, involving {} different nodes".format(
    test_data.n_interactions, test_data.n_unique_nodes))

  return node_features, edge_features, full_data, train_data, test_data
         


def compute_time_statistics(sources, destinations, timestamps):
  last_timestamp_sources = dict()
  last_timestamp_dst = dict()
  all_timediffs_src = []
  all_timediffs_dst = []
  for k in range(len(sources)):
    source_id = sources[k]
    dest_id = destinations[k]
    c_timestamp = timestamps[k]
    if source_id not in last_timestamp_sources.keys():
      last_timestamp_sources[source_id] = 0
    if dest_id not in last_timestamp_dst.keys():
      last_timestamp_dst[dest_id] = 0
    all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
    all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
    last_timestamp_sources[source_id] = c_timestamp
    last_timestamp_dst[dest_id] = c_timestamp
  assert len(all_timediffs_src) == len(sources)
  assert len(all_timediffs_dst) == len(sources)
  mean_time_shift_src = np.mean(all_timediffs_src)
  std_time_shift_src = np.std(all_timediffs_src)
  mean_time_shift_dst = np.mean(all_timediffs_dst)
  std_time_shift_dst = np.std(all_timediffs_dst)

  return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst
