from typing import NamedTuple, List, Optional

# hybridize fails for now
hybridize = False


class Hyperparams(NamedTuple):
    
    #Paramètres à modifier 
    path_folder = "."
    num_cells: int = 40
    learning_rate: float = 1e-3
    modele: str =  "deepnegpol"
    #Pour DeepNegPol seulement
    num_cells1: int = 20
    num_cells2: int = 40

    batch_size: int = 16
    num_batches_per_epoch: int = 10
    epochs: int = 1
    num_eval_samples: int = 10
    num_layers: int = 2
    rank: int = 8
    conditioning_length: int = 50
    target_dim_sample: int = 5
    dropout_rate: float = 0.01
    patience: int = 50
    cell_type: str = "lstm"
    hybridize: bool = True
    
    
    #Paramètres fixes modele
    pred_days: int = 1
    lags_seq: Optional[List[int]] = [1,2,4,12,24,48]
    given_days = 3*pred_days
    


    
    
    

class FastHyperparams(NamedTuple):
    p = Hyperparams()
    epochs: int = 1
    num_batches_per_epoch: int = 1
    num_cells: int = 1
    num_layers: int = 1
    num_eval_samples: int = 1
    modele: str = "DeepNegPol"
    cell_type: str = "lstm"
    conditioning_length: int = 10
    batch_size: int = 16
    rank: int = 5

    target_dim_sample: int = p.target_dim_sample
    patience: int = p.patience
    hybridize: bool = hybridize
    learning_rate: float = p.learning_rate
    dropout_rate: float = p.dropout_rate
    lags_seq: Optional[List[int]] = p.lags_seq
    #scaling: bool = p.scaling


if __name__ == '__main__':
    params = Hyperparams()
    print(repr(params))
