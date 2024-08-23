import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200, vs=None, rs=None):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)

  val_ap, val_auc = [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch-1):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]
      true_label = data.labels[s_idx:e_idx]
      sources_batch = torch.tensor(sources_batch)
      destinations_batch = torch.tensor(destinations_batch)
      timestamps_batch = torch.tensor(timestamps_batch)
      
      
      
      pos_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                      timestamps_batch, 
                                                      edge_idxs_batch, n_neighbors,update_memory=True,
                                                      next_R=model.R + rs[k].to_dense().to(model.device),
                                                      next_V=model.V + vs[k].to_dense().to(model.device))
      
      pred_score = (pos_prob).cpu().numpy()
      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))
  return np.mean(val_ap), np.mean(val_auc), vs, rs


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
  pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding = tgn.compute_temporal_embeddings(sources_batch, destinations_batch,
                                                                                timestamps_batch, edge_idxs_batch,
                                                                                n_neighbors, persist=False,
                                                                                nodel_level=True)
      pred_prob_batch = decoder(source_embedding).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

  auc_roc = roc_auc_score(data.labels, pred_prob)
  return auc_roc
