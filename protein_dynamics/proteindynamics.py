import torch.nn as nn
import torch
from .icoord import get_chi_angles, get_backbone_torsions
from .common_layers import PerResidueEncoder, ResiduePairEncoder, GAEncoder, BiGRUProteinModel

#test

class ProteinDynamic(nn.Module):
    def __init__(self, cfg):
        super(ProteinDynamic, self).__init__()
        self.cfg = cfg

        self.single_encoder = PerResidueEncoder(
            feat_dim=cfg.encoder.node_feat_dim,
            max_num_atoms=5,    # N, CA, C, O, CB,
        )
        self.masked_bias = nn.Embedding(
            num_embeddings = 2,
            embedding_dim = cfg.encoder.node_feat_dim,
            padding_idx = 0,
        )
        self.pair_encoder = ResiduePairEncoder(
            feat_dim=cfg.encoder.pair_feat_dim,
            max_num_atoms=5,    # N, CA, C, O, CB,
        )
        self.attn_encoder = GAEncoder(**cfg.encoder)
        
        self.encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        
        """
        Use the GRU to predict the next time step protein graph embeddings, edied by Yaoyao
        """
        self.temporal_model = BiGRUProteinModel(input_dim = cfg.encoder.node_feat_dim,
                                                hidden_dim = 128, 
                                                output_dim = 128, 
                                                time_emb_size = 128 , 
                                                max_seq_len = cfg.encoder.max_protein_size,  ##not sure max_protein_sequence's name in config
                                                num_layers= 2, 
                                                dropout=0.2)

        self.decoder = decoder(**cfg.encoder)
        
        
    def forward(self, x, t):
        
        """
        Edited by Yaoyao
        """
        encoded_graph = self.encoder(x)
        next_time_step_graph = self.temporal_model(encoded_graph, t)
        
        decoded_graph = self.decoder(next_time_step_graph)
        
        return decoded_graph
    