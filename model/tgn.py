import logging
import numpy as np
import torch
from collections import defaultdict

from utils.utils import MergeLayer
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode


class TGN(torch.nn.Module):
  def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
            use_memory=False,
               memory_update_at_start=True, message_dimension=100,
               memory_dimension=500,
               message_function="mlp",
               mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
               std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
               memory_updater_type="gru",
               use_destination_embedding_in_message=False,
               use_source_embedding_in_message=False,
               beta=0.1,
               r_dim=4,
               Generator = None):
    super(TGN, self).__init__()

    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device
    self.logger = logging.getLogger(__name__)

    self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
    self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

    self.n_node_features = self.node_raw_features.shape[1]
    self.n_nodes = self.node_raw_features.shape[0]
    self.n_edge_features = self.edge_raw_features.shape[1]
    self.embedding_dimension = self.n_node_features
    self.n_neighbors = n_neighbors
    self.use_destination_embedding_in_message = use_destination_embedding_in_message
    self.use_source_embedding_in_message = use_source_embedding_in_message
    self.beta = beta

    self.use_memory = use_memory
    self.time_encoder = TimeEncode(dimension=self.n_node_features)
    self.memory = None

    self.Generator = Generator

    self.mean_time_shift_src = mean_time_shift_src
    self.std_time_shift_src = std_time_shift_src
    self.mean_time_shift_dst = mean_time_shift_dst
    self.std_time_shift_dst = std_time_shift_dst

    ####### Positional features #######
    self.r_dim = r_dim
    self.R = torch.zeros((node_features.shape[0],node_features.shape[0], r_dim), requires_grad=False).to(device)
    self.P = torch.zeros((r_dim, r_dim)).to(device)
    self.P[1:, :-1] = torch.eye(r_dim-1, requires_grad=False).to(device)
    for i in range(node_features.shape[0]):
        self.R[i, i, 0] = 1.0
    self.V = torch.eye(node_features.shape[0], requires_grad=False).to(device)
    ############################

    if self.use_memory:
      self.memory_dimension = memory_dimension
      self.memory_update_at_start = memory_update_at_start
      raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                              self.time_encoder.dimension
      message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
      self.memory = Memory(n_nodes=self.n_nodes,
                           memory_dimension=self.memory_dimension,
                           input_dimension=message_dimension,
                           message_dimension=message_dimension,
                           device=device)
      self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                       device=device)
      self.message_function = get_message_function(module_type=message_function,
                                                   raw_message_dimension=raw_message_dimension,
                                                   message_dimension=message_dimension)
      self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                               memory=self.memory,
                                               message_dimension=message_dimension,
                                               memory_dimension=self.memory_dimension,
                                               device=device)

    self.embedding_module = get_embedding_module(module_type="GADY",
                                                 node_features=self.node_raw_features,
                                                 edge_features=self.edge_raw_features,
                                                 neighbor_finder=self.neighbor_finder,
                                                 time_encoder=self.time_encoder,
                                                 n_layers=self.n_layers,
                                                 n_node_features=self.n_node_features,
                                                 n_edge_features=self.n_edge_features,
                                                 n_time_features=self.n_node_features,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 use_memory=use_memory,
                                                 n_neighbors=self.n_neighbors, beta=self.beta,
                                                 r_dim=r_dim)

    # MLP to compute probability on an edge given two node embeddings
    self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                     self.n_node_features, 1)

  def reset_VR(self):
      self.R = torch.zeros(self.R.shape, requires_grad=False).to(self.device)
      for i in range(self.R.shape[0]):
          self.R[i, i, 0] = 1.0
      self.V = torch.eye(self.R.shape[0], requires_grad=False).to(self.device)

  def data_collect(self,source_nodes, timestamps, n_layers, n_neighbors = 20): 
    source_nodes = torch.from_numpy(source_nodes).long().to(self.device)
    timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

    memory = self.memory.get_memory(list(range(self.n_nodes)))
    source_node_features = self.embedding_module.node_features[source_nodes, :] + memory[source_nodes, :]
    
    neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
          source_nodes,  
          timestamps,  
          n_neighbors=n_neighbors)
    neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
    edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)
    edge_deltas = timestamps[:, np.newaxis] - edge_times
    edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)  
    
    neighbors_features = self.embedding_module.node_features[neighbors, :] + memory[neighbors, :] 
    nei_edge_features = self.embedding_module.edge_features[edge_idxs, :]
    
    
    mask = neighbors_torch == 0
    src_idx = source_nodes 
    neigh_idx = neighbors_torch 
    
    
    return source_node_features, timestamps_torch, neighbors_features, nei_edge_features,edge_deltas_torch, mask, src_idx,  neigh_idx, neighbors

  def compute_neg_temporal_embeddings(self, source_nodes, destination_nodes, edge_times,
                                  edge_idxs, n_neighbors=20, nograd = False):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """
    n_samples = len(source_nodes)
    source_node_features = list(range(self.n_layers))    # 源节点
    neighbor_embeddings = list(range(self.n_layers)) 
    edge_features = list(range(self.n_layers)) 
    mask = list(range(self.n_layers)) 
    timestamps_nodes = list(range(self.n_layers)) 
    time_diffs_nodes = list(range(self.n_layers)) 
    src_idx = list(range(self.n_layers)) 
    neigh_idx = list(range(self.n_layers)) 
    edge_gaps = list(range(self.n_layers)) 
    roots = list(range(self.n_layers)) 
    targets = list(range(self.n_layers)) 
    
    for i in range(self.n_layers):
      if i == 0:
        n_samples = len(source_nodes)
        nodes = np.concatenate([source_nodes, destination_nodes])
        targets[i] = np.concatenate([destination_nodes, source_nodes])
        roots[i] = torch.from_numpy(nodes).long().to(self.device)
        targets[i] = torch.from_numpy(targets[i]).long().to(self.device)
        # positives = np.concatenate([source_nodes, destination_nodes])
        timestamps = np.concatenate([edge_times, edge_times])

        memory = None
        time_diffs = None
        if self.use_memory:
          if self.memory_update_at_start:
            # Update memory for all nodes with messages stored in previous batches
            memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                          self.memory.messages)
          else:
            memory = self.memory.get_memory(list(range(self.n_nodes)))
            last_update = self.memory.last_update

          ### Compute differences between the time the memory of a node was last updated,
          ### and the time for which we want to compute the embedding of a node
          source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
            source_nodes].long()
          source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
          destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
            destination_nodes].long()
          destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

          time_diffs = torch.cat([source_time_diffs, destination_time_diffs],
                                dim=0)
          time_diffs_nodes[i] = time_diffs
          #找出对应的邻居，并且把相关信息都整合起来
        roots[i] = roots[i].expand(n_neighbors, roots[i].shape[0]).T.flatten()
        targets[i] = targets[i].expand(n_neighbors, targets[i].shape[0]).T.flatten()
        source_node_features[i], timestamps_nodes[i],neighbor_embeddings[i],edge_features[i],edge_gaps[i], mask[i], src_idx[i],neigh_idx[i], neighbors = self.data_collect(nodes, timestamps, i,n_neighbors= n_neighbors)
      else:
        nodes = neighbors.flatten()
        timestamps = np.repeat(timestamps, n_neighbors)
        roots[i] = roots[i-1]
        targets[i] = targets[i-1]
        roots[i] = roots[i].expand(n_neighbors, roots[i].shape[0]).T.flatten()
        targets[i] = targets[i].expand(n_neighbors, targets[i].shape[0]).T.flatten()
        
        source_node_features[i], timestamps_nodes[i],neighbor_embeddings[i],edge_features[i],edge_gaps[i], mask[i], src_idx[i],neigh_idx[i], neighbors = self.data_collect(nodes, timestamps, i,n_neighbors= n_neighbors)
    
    neg_source_node_features = list(range(self.n_layers)) 
    neg_neighbor_embeddings = list(range(self.n_layers)) 
    neg_edge_features = list(range(self.n_layers)) 
    neg_mask = list(range(self.n_layers)) 
    neg_timestamps_nodes = list(range(self.n_layers)) 
    neg_time_diffs_nodes = list(range(self.n_layers)) 
    neg_src_struc = list(range(self.n_layers)) 
    neg_neigh_struc = list(range(self.n_layers)) 
    neg_edge_gaps = list(range(self.n_layers)) 
    neg_roots = list(range(self.n_layers)) 
    neg_targets = list(range(self.n_layers)) 
    
    for i in range(self.n_layers-1, - 1, -1):
      if i == 1:
        neg_source_node_features[i],neg_neighbor_embeddings[i], neg_edge_features[i], neg_edge_gaps[i] = self.Generator(
                                                                              source_node_features = source_node_features[i],
                                                                              timestamps_nodes = timestamps_nodes[i],
                                                                              neighbor_embeddings = neighbor_embeddings[i].reshape(neighbor_embeddings[i].shape[0],-1),
                                                                              edge_features = edge_features[i].reshape(edge_features[i].shape[0],-1),
                                                                              # time_diffs_nodes = time_diffs_nodes[i],
                                                                              edge_gaps = edge_gaps[i],
                                                                              layer = i
                                                                              )
        if nograd:
          source_embedding = self.embedding_module.aggregate(n_layer = i,source_node_features = neg_source_node_features[i].detach(),
                                          neighbor_embeddings = neg_neighbor_embeddings[i].detach(), edge_features = neg_edge_features[i].detach(),
                                          mask = mask[i].detach(), timestamps = neg_edge_gaps[i].detach(), V = self.V, R = self.R, src_idx = src_idx[i],
                                        neigh_idx  = neigh_idx[i], roots = roots[i],targets=targets[i])
        else:
          source_embedding = self.embedding_module.aggregate(n_layer = i,source_node_features = neg_source_node_features[i],
                                          neighbor_embeddings = neg_neighbor_embeddings[i], edge_features = neg_edge_features[i],
                                          mask = mask[i], timestamps = neg_edge_gaps[i], V = self.V, R = self.R, src_idx = src_idx[i],
                                        neigh_idx  = neigh_idx[i], roots = roots[i],targets=targets[i])
      
      elif i == 0:
        neg_source_node_features[i],neg_neighbor_embeddings[i], neg_edge_features[i],neg_edge_gaps[i]  = self.Generator(
                                                                      source_node_features = source_node_features[i],
                                                                      timestamps_nodes = timestamps_nodes[i],
                                                                      neighbor_embeddings = neighbor_embeddings[i].reshape(neighbor_embeddings[i].shape[0],-1),
                                                                      edge_features = edge_features[i].reshape(edge_features[i].shape[0],-1),
                                                                      # time_diffs_nodes[i],
                                                                      edge_gaps = edge_gaps[i],
                                                                      layer = i,
                                                                      before_embedding = source_embedding.reshape(source_node_features[i].shape[0],-1))
        if nograd:
          source_embedding = self.embedding_module.aggregate(n_layer = i,source_node_features = neg_source_node_features[i].detach(),
                                          neighbor_embeddings = neg_neighbor_embeddings[i].detach(), edge_features = neg_edge_features[i].detach(),
                                          mask = mask[i].detach(), timestamps = neg_edge_gaps[i].detach(), V = self.V, R = self.R, src_idx = src_idx[i],
                                        neigh_idx  = neigh_idx[i], roots = roots[i],targets=targets[i])
        else:
          source_embedding = self.embedding_module.aggregate(n_layer = i,source_node_features = neg_source_node_features[i],
                                          neighbor_embeddings = neg_neighbor_embeddings[i], edge_features = neg_edge_features[i],
                                          mask = mask[i], timestamps = neg_edge_gaps[i], V = self.V, R = self.R, src_idx = src_idx[i],
                                        neigh_idx  = neigh_idx[i], roots = roots[i],targets=targets[i])
    source_node_embedding = source_embedding[:n_samples]
    destination_node_embedding = source_embedding[n_samples: 2 * n_samples]
    neg_source_node_features[1] = []
    neg_neighbor_embeddings[1] = []
    neg_edge_features[1] = []
    neg_edge_gaps[1] = []
    
        
    return source_node_embedding, destination_node_embedding, neg_source_node_features[0],neg_neighbor_embeddings[0],neg_edge_features[0],neg_edge_gaps[0]
  def compute_temporal_embeddings(self, source_nodes, destination_nodes, edge_times,
                                  edge_idxs, n_neighbors=20, persist=True, nodel_level=False):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """

    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes])
    targets = np.concatenate([destination_nodes, source_nodes])
    positives = np.concatenate([source_nodes, destination_nodes])
    timestamps = np.concatenate([edge_times, edge_times])

    memory = None
    time_diffs = None
    if self.use_memory:
      if self.memory_update_at_start:
        # Update memory for all nodes with messages stored in previous batches
        memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                      self.memory.messages)
      else:
        memory = self.memory.get_memory(list(range(self.n_nodes)))
        last_update = self.memory.last_update
    targets = torch.from_numpy(targets).long().to(self.device)
    node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors,
                                                             V=self.V, R=self.R, targets=targets,
                                                             node_level=nodel_level)

    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]

    if self.use_memory and persist:
      if self.memory_update_at_start:
        self.update_memory(positives, self.memory.messages)

        self.memory.clear_messages(positives)

      unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                    source_node_embedding,
                                                                    destination_nodes,
                                                                    destination_node_embedding,
                                                                    edge_times, edge_idxs)
      unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                              destination_node_embedding,
                                                                              source_nodes,
                                                                              source_node_embedding,
                                                                              edge_times, edge_idxs)
      if self.memory_update_at_start:
        self.memory.store_raw_messages(unique_sources, source_id_to_messages)
        self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
      else:
        self.update_memory(unique_sources, source_id_to_messages)
        self.update_memory(unique_destinations, destination_id_to_messages)

    return source_node_embedding, destination_node_embedding
  
  def compute_neg_edge_probabilities(self, source_nodes, destination_nodes, edge_times,
                                 edge_idxs, n_neighbors=20, update_memory = False,next_R=None, next_V=None,nograd = False):
    """
    Compute probabilities for negative edges between sources and destination by first computing temporal embeddings using the TGN encoder and then feeding them
    into the MLP decoder.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Probabilities for both the positive and negative edges
    """
    neg_samples = list(range(4)) 
    source_embedding, destination_embedding,neg_samples[0],neg_samples[1],neg_samples[2],neg_samples[3] = self.compute_neg_temporal_embeddings(source_nodes, destination_nodes,
                                                                                     edge_times, edge_idxs, n_neighbors,
                                                                                     nograd=nograd)
    score = self.affinity_score(source_embedding, destination_embedding)
    self.V, self.R = next_V, next_R
    return score.sigmoid(), neg_samples
  def compute_edge_probabilities(self, source_nodes, destination_nodes, edge_times,
                                 edge_idxs, n_neighbors=20, update_memory = False,next_R=None, next_V=None):
    """
    Compute probabilities for edges between sources and destination by first computing temporal embeddings using the TGN encoder and then feeding them
    into the MLP decoder.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Probabilities for both the positive and negative edges
    """
    source_embedding, destination_embedding = self.compute_temporal_embeddings(source_nodes, destination_nodes,
                                                                                     edge_times, edge_idxs, n_neighbors,
                                                                                     persist=update_memory)
    score = self.affinity_score(source_embedding, destination_embedding)
    self.V, self.R = next_V, next_R
    return score.sigmoid()

  def update_memory(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    # Update the memory with the aggregated messages
    self.memory_updater.update_memory(unique_nodes, unique_messages,
                                      timestamps=unique_timestamps)

  def get_updated_memory(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                 unique_messages,
                                                                                 timestamps=unique_timestamps)

    return updated_memory, updated_last_update

  def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs):
    edge_times = edge_times.float().to(self.device)
    edge_features = self.edge_raw_features[edge_idxs]

    source_memory = self.memory.get_memory(source_nodes) if not \
      self.use_source_embedding_in_message else source_node_embedding
    destination_memory = self.memory.get_memory(destination_nodes) if \
      not self.use_destination_embedding_in_message else destination_node_embedding

    source_time_delta = edge_times - self.memory.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1)

    source_message = torch.cat([source_memory, destination_memory, edge_features,
                                source_time_delta_encoding],
                               dim=1)
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)

    for i in range(len(source_nodes)):
      messages[source_nodes[i]].append((source_message[i], edge_times[i]))

    return unique_sources, messages

  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder
