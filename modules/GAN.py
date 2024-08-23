import torch
import torch.nn as nn
       
        
class Generator(nn.Module):
    def __init__(self, node_features_dim=172,edge_time_dim = 1, n_neighbors=20, nei_embedding_dim = 172,edge_features_dim = 172,
                 time_diff_dim=1, edge_time_gaps_dim = 1, batch_size=200,device = "cpu"):   ##edge idx maybe not need
        super().__init__() 
        self.batch_size = batch_size
        self.node_features_dim = node_features_dim
        self.edge_time_dim = edge_time_dim
        self.n_neighbors = n_neighbors 
        self.nei_embedding_dim = nei_embedding_dim 
        self.edge_features_dim = edge_features_dim
        self.time_diff_dim = time_diff_dim
        self.edge_time_gaps_dim = edge_time_gaps_dim
        self.layer0_dim = self.node_features_dim + self.edge_time_dim + n_neighbors*(self.nei_embedding_dim +self.edge_features_dim+self.nei_embedding_dim+self.edge_time_gaps_dim)
        self.layer1_dim = self.node_features_dim + self.edge_time_dim + n_neighbors*(self.nei_embedding_dim + self.edge_features_dim+self.edge_time_gaps_dim)
        self.device = device
        self.out_dim = self.node_features_dim + self.n_neighbors*(self.nei_embedding_dim + self.edge_features_dim + self.edge_time_gaps_dim)
        
        self.time_emb_dim = 200
        self.layer0 = nn.Sequential(
            nn.Linear(2*self.layer0_dim, 2*self.layer0_dim),  
            nn.LeakyReLU(0.2),  
            nn.Linear(2*self.layer0_dim, self.layer0_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.layer0_dim, self.layer0_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.layer0_dim, 2*self.layer0_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(2*self.layer0_dim, self.out_dim),        
            nn.Tanh() 
            
        )
        self.layer1 = nn.Sequential(
            nn.Linear(2*self.layer1_dim, 2*self.layer1_dim), 
            nn.LeakyReLU(0.2), 
            nn.Linear(2*self.layer1_dim, self.layer1_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.layer1_dim, self.layer1_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.layer1_dim, 2*self.layer1_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(2*self.layer1_dim, self.out_dim),            
            nn.Tanh()
        )


    def forward(self, source_node_features,timestamps_nodes,neighbor_embeddings, edge_features, edge_gaps,layer,before_embedding = None):
        if layer == 0:
            self.noise = (torch.randn([source_node_features.shape[0],self.layer0_dim])-0.5)*45
            self.noise = self.noise.to(self.device)
            input =  torch.cat([source_node_features,timestamps_nodes,neighbor_embeddings,edge_features,before_embedding,edge_gaps,self.noise],dim=1)
            output = self.layer0(input)
            final = (output+1)/2
        else:
            self.noise = (torch.randn([source_node_features.shape[0],self.layer1_dim])-0.5)*45
            self.noise = self.noise.to(self.device)
            input =  torch.cat([source_node_features,timestamps_nodes,neighbor_embeddings,edge_features,edge_gaps,self.noise],dim=1)
            output = self.layer1(input)
            final = (output+1)/2
        return final[:,:self.node_features_dim]*(torch.max(source_node_features)-torch.min(source_node_features)), final[:,self.node_features_dim:self.node_features_dim+self.n_neighbors*self.nei_embedding_dim].reshape(final.shape[0],self.n_neighbors,-1)*(torch.max(neighbor_embeddings)-torch.min(neighbor_embeddings)), \
            final[:,self.node_features_dim+self.n_neighbors*self.nei_embedding_dim:self.node_features_dim+self.n_neighbors*(self.nei_embedding_dim+self.edge_features_dim)].reshape(final.shape[0],self.n_neighbors,-1)*(torch.max(edge_features)-torch.min(edge_features)), \
                final[:,self.node_features_dim+self.n_neighbors*(self.nei_embedding_dim+self.edge_features_dim):self.node_features_dim+self.n_neighbors*(self.nei_embedding_dim+self.edge_features_dim+self.edge_time_gaps_dim)].reshape(final.shape[0],self.n_neighbors)*(torch.max(edge_gaps)-torch.min(edge_gaps))