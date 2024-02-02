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
    num_batches_per_epoch: int = 100
    epochs: int = 100
    num_eval_samples: int = 100
    num_layers: int = 2
    rank: int = 10
    conditioning_length: int = 100
    target_dim_sample: int = 20
    dropout_rate: float = 0.01
    patience: int = 50
    cell_type: str = "lstm"
    hybridize: bool = True
    
    
    #Paramètres fixes modele
    pred_days: int = 1
    lags_seq1: Optional[List[int]] = [1,7,14]
    lags_seq2: Optional[List[int]] = [1, 2, 4, 12, 24, 48]
    lags_seq3: Optional[List[int]] = [1, 24,168]
    lags_seq4: Optional[List[int]] = [1, 4, 96]
    


if __name__ == '__main__':
    params = Hyperparams()
    print(repr(params))
