from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.torch.modules.distribution_output import DistributionOutput
from pts.model import weighted_average
from pts.modules import MeanScaler, NOPScaler, FeatureEmbedder

import numpy as np

from scipy.stats import nbinom



class DeepNEGPOLTrainingNetwork(nn.Module):
    @validated()
    def __init__(
        self,
        input_size1: int,
        input_size2: int,
        num_layers1: int,
        num_layers2: int,
        num_cells1: int,
        num_cells2: int,
        history_length: int,
        context_length: int,
        prediction_length: int,
        dropout_rate: float,
        lags_seq: List[int],
        target_dim: int,
        cardinality: List[int],
        embedding_dimension: List[int],
        scaling: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_size1 = input_size1
        self.input_size2 = input_size2
        self.num_layers1 = num_layers1
        self.num_layers2 = num_layers2
        self.num_cells1 = num_cells1
        self.num_cells2 = num_cells2
        self.history_length = history_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.num_cat = len(cardinality)
        self.target_dim = target_dim
        self.scaling = scaling
        self.target_dim_sample = target_dim

        assert len(set(lags_seq)) == len(lags_seq), "no duplicated lags allowed!"
        lags_seq.sort()

        self.lags_seq = lags_seq


        self.target_dim = target_dim

     
        self.lstm1 = nn.LSTM(input_size=input_size1,
                    hidden_size=num_cells1,
                    num_layers=num_layers1,
                    bias=True,
                    batch_first=True,
                    dropout=dropout_rate)

        self.lstm2 = nn.LSTM(input_size=input_size2,
                    hidden_size=num_cells2,
                    num_layers=num_layers2,
                    bias=True,
                    batch_first=True,
                    dropout=dropout_rate)
        
        
        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm1._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm1, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)
        
        for names in self.lstm2._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm2, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)
        

        
        #self.perturbation_linear = nn.Linear(8, 50)
        self.softplus = nn.Softplus()
        self.distribution_mu = nn.Linear(num_cells1, 1)
        self.distribution_sigma = nn.Linear(num_cells1, 1)
        self.distribution_alpha = nn.Linear(num_cells2, self.target_dim)
        self.EL = Exponential_Linear()
        self.linear_pert = nn.Linear(8, 25) 
        self.sigmoid = nn.Sigmoid()


        #self.embed = FeatureEmbedder(
        #    cardinalities=cardinality, embedding_dims=embedding_dimension
        #)

        if scaling:
            self.scaler = MeanScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

    @staticmethod
    def get_lagged_subsequences(
        sequence: torch.Tensor,
        sequence_length: int,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence.
        Parameters
        ----------
        sequence
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        sequence_length
            length of sequence in the T (time) dimension (axis = 1).
        indices
            list of lag indices to be used.
        subsequences_length
            length of the subsequences to be extracted.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I),
            where S = subsequences_length and I = len(indices),
            containing lagged subsequences.
            Specifically, lagged[i, :, j, k] = sequence[i, -indices[k]-S+j, :].
        """
        # we must have: history_length + begin_index >= 0
        # that is: history_length - lag_index - sequence_length >= 0
        # hence the following assert
        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag "
            f"{max(indices)} while history length is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...].unsqueeze(1))
        return torch.cat(lagged_values, dim=1).permute(0, 2, 3, 1)



    def unroll(
        self,
        lags1: torch.Tensor,
        lags2: torch.Tensor,
        scale: torch.Tensor,
        time_feat: torch.Tensor,
        unroll_length: int,
        begin_state1: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        begin_state2: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:

        #Récupération de la valeur de scaling tot
        scaleT = scale.sum(dim=2)
            
        # (batch_size, sub_seq_len, 1, num_lags)
        lags1_scaled = lags1 / scaleT.unsqueeze(-1)
        
        input_lags1 = lags1_scaled.reshape(
            (-1, unroll_length, len(self.lags_seq))
        )
        
        # (batch_size, sub_seq_len, target_dim, num_lags)
        lags2_scaled = lags2 / scale.unsqueeze(-1)

        input_lags2 = lags2_scaled.reshape(
            (-1, unroll_length, len(self.lags_seq) * self.target_dim)
        )


        # (batch_size, sub_seq_len, input1_dim)

        inputs1 = torch.cat((input_lags1, time_feat), dim=-1)
        # (batch_size, sub_seq_len, input2_dim)
        inputs2 = torch.cat((input_lags2, time_feat), dim=-1)

        # unroll encoder
        outputs1, state1 = self.lstm1(inputs1, begin_state1)
        outputs2, state2 = self.lstm2(inputs2, begin_state2)



        # assert_shape(outputs, (-1, unroll_length, self.num_cells))
        # for s in state:
        #     assert_shape(s, (-1, self.num_cells))

        # assert_shape(
        #     lags_scaled, (-1, unroll_length, self.target_dim, len(self.lags_seq)),
        # )

        return outputs1, state1, lags1_scaled, inputs1, outputs2, state2, lags2_scaled, inputs2



    def unroll_encoder(
        self,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: Optional[torch.Tensor],
        future_target_cdf: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Unrolls the RNN encoder over past and, if present, future data.
        Returns outputs and state of the encoder, plus the scale of
        past_target_cdf and a vector of static features that was constructed
        and fed as input to the encoder. All tensor arguments should have NTC
        layout.

        Parameters
        ----------

        past_time_feat
            Past time features (batch_size, history_length, num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)
        future_target_cdf
            Future marginal CDF transformed target values (batch_size,
            prediction_length, target_dim)


        Returns
        -------
        outputs
            RNN outputs (batch_size, seq_len, num_cells)
        states
            RNN states. Nested list with (batch_size, num_cells) tensors with
        dimensions target_dim x num_layers x (batch_size, num_cells)
        scale
            Mean scales for the time series (batch_size, 1, target_dim)
        lags_scaled
            Scaled lags(batch_size, sub_seq_len, target_dim, num_lags)
        inputs
            inputs to the RNN

        """

        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        #Préparer Inputs pour les cellules LSTM
        if future_time_feat is None or future_target_cdf is None:
            time_feat = past_time_feat[:, -self.context_length :, ...]
            sequence = past_target_cdf
            sequence_length = self.history_length
            subsequences_length = self.context_length
        else:
            time_feat = torch.cat(
                (past_time_feat[:, -self.context_length :, ...], future_time_feat,),
                dim=1,
            )
            sequence = torch.cat((past_target_cdf, future_target_cdf), dim=1)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        #Prise en compte d'une fenêtre depuis lag -1 et une autre fenêtre depuis lag - 336 (1 semaine)
        # (batch_size, sub_seq_len, target_dim, num_lags)
        lags2 = self.get_lagged_subsequences(
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        )
        
        # (batch_size, sub_seq_len, 1, num_lags)
        lags1 = lags2.sum(2)

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, target_dim)
        _, scale = self.scaler(
            past_target_cdf[:, -self.context_length :, ...],
            past_observed_values[:, -self.context_length :, ...],
        )

        outputs1, state1, lags1_scaled, inputs1, outputs2, state2, lags2_scaled, inputs2 = self.unroll(
            lags1=lags1,
            lags2=lags2,
            scale = scale,
            time_feat=time_feat,
            unroll_length=subsequences_length,
            begin_state1=None,
            begin_state2=None,
        )

        return outputs1, state1, lags1_scaled, inputs1, outputs2, state2, lags2_scaled, inputs2, scale





    def forward(
        self,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target_cdf: torch.Tensor,
        future_observed_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Computes the loss for training DeepVAR, all inputs tensors representing
        time series have NTC layout.

        Parameters
        ----------

        past_time_feat
            Dynamic features of past time series (batch_size, history_length,
            num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)
        future_target_cdf
            Future marginal CDF transformed target values (batch_size,
            prediction_length, target_dim)
        future_observed_values
            Indicator whether or not the future values were observed
            (batch_size, prediction_length, target_dim)

        Returns
        -------
        distr
            Loss with shape (batch_size, 1)
        likelihoods
            Likelihoods for each time step
            (batch_size, context + prediction_length, 1)
        distr_args
            Distribution arguments (context + prediction_length,
            number_of_arguments)
        """

        seq_len = self.context_length + self.prediction_length
        
        # unroll the decoder in "training mode", i.e. by providing future data
        # as well
        rnn_outputs1, _, _, inputs1, rnn_outputs2, _, _, inputs2, scale = self.unroll_encoder(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,
            future_target_cdf=future_target_cdf,
        )

        # put together target sequence
        # (batch_size, seq_len, target_dim)
        target = torch.cat(
            (past_target_cdf[:, -self.context_length :, ...], future_target_cdf), dim=1,
        )

        
        subsequences_length = self.context_length + self.prediction_length
        batch_size = rnn_outputs1.shape[0]

        
        #likelihoods = torch.zeros(batch_size, subsequences_length)
        #distr_args = torch.zeros(subsequences_length, 2 + self.target_dim)
        
        target = target.permute(1, 0, 2)
        targetT = target.sum(2)
        
        scaleT = scale.sum(2)
        
        loss_l = torch.zeros(subsequences_length)
        for t in range(subsequences_length):
            #rnn_outputs1_permute = rnn_outputs1[:,t,:].permute(0, 2, 1).contiguous().view(rnn_outputs1.shape[0], -1)
            #rnn_outputs2_permute = rnn_outputs2[:,t,:].permute(0, 2, 1).contiguous().view(rnn_outputs1.shape[0], -1)
      
            mu = self.softplus(self.distribution_mu(rnn_outputs1[:,t,:]))
            sigma = self.softplus(self.distribution_sigma(rnn_outputs1[:,t,:]))  + 1e-6
            alpha = torch.squeeze(self.EL(self.distribution_alpha(rnn_outputs2[:,t,:])))

            #rescaling 
            mu = torch.squeeze(mu * scaleT)
            sigma = torch.squeeze(sigma*torch.sqrt(scaleT))
            
            #likelihood calculation
            #likelihoods[:,t] = -loss_fn(mu, sigma, alpha, targetT[t], target[t])
            loss_l[t] = loss_fn(mu, sigma, alpha, targetT[t], target[t])



        # we sum the last axis to have the same shape for all likelihoods
        # likelihoods : (batch_size, subseq_length)
        

        return torch.mean(loss_l)
    
    
    

class Exponential_Linear(nn.Module):
    def __init__(self):
        super(Exponential_Linear, self).__init__()
    
    def forward(self, x):
        x[x>=0] = x[x>=0] + 1
        x[x<0] = torch.exp(x[x<0])
        return x
    


def loss_fn(mu, sigma, alpha, labelT, label):

    if len(labelT.shape)==1:   
            zero_index = (labelT != 0)
            
            LL_nb = torch.lgamma(labelT[zero_index] + sigma[zero_index]) - torch.lgamma(labelT[zero_index] + 1) - torch.lgamma(sigma[zero_index]) \
                + sigma[zero_index] * torch.log(sigma[zero_index] / (sigma[zero_index]+mu[zero_index])) \
                + labelT[zero_index] * torch.log(mu[zero_index] / (sigma[zero_index] + mu[zero_index]))
           
          
            LL_dm = (torch.lgamma(torch.sum(alpha[zero_index],1)) + torch.lgamma((label.sum(axis=1))[zero_index]+1)) - torch.lgamma((label.sum(axis=1))[zero_index]+torch.sum(alpha[zero_index],1)) \
                    + torch.sum((torch.lgamma(label[zero_index]+alpha[zero_index]) - (torch.lgamma(alpha[zero_index]) + torch.lgamma(label[zero_index]+1) ) ),1 )
              
    if len(labelT.shape)==2:
            LL_nb = torch.sum( torch.lgamma(labelT + sigma) - torch.lgamma(labelT + 1) - torch.lgamma(sigma) \
                + sigma * torch.log(sigma / (sigma+mu)) \
                + labelT * torch.log(mu / (sigma + mu)), axis=1 )
              
            LL_dm = torch.sum( (torch.lgamma(torch.sum(alpha,2)) + torch.lgamma((label.sum(axis=2))+1)) - torch.lgamma((label.sum(axis=2))+torch.sum(alpha,2)) \
                    + torch.sum((torch.lgamma(label+alpha) - (torch.lgamma(alpha) + torch.lgamma(label+1) ) ),2 ), axis=1)
   
    LL = LL_nb + LL_dm    
    return -torch.mean(LL)






class DeepNEGPOLPredictionNetwork(DeepNEGPOLTrainingNetwork):
    def __init__(self, num_parallel_samples: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples

        # for decoding the lags are shifted by one,
        # at the first time-step of the decoder a lag of one corresponds to
        # the last target value
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    def sampling_decoder(
        self,
        past_target_cdf: torch.Tensor,
        time_feat: torch.Tensor,
        scale: torch.Tensor,
        begin_states1: Union[List[torch.Tensor], torch.Tensor],
        begin_states2: Union[List[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes sample paths by unrolling the RNN starting with a initial
        input and state.

        Parameters
        ----------
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)


        time_feat
            Dynamic features of future time series (batch_size, history_length,
            num_features)
        scale
            Mean scale for each time series (batch_size, 1, target_dim)
        begin_states1, begin_states2
            List of initial states for the RNN layers (batch_size, num_cells)
        Returns
        --------
        sample_paths : Tensor
            A tensor containing sampled paths. Shape: (1, num_sample_paths,
            prediction_length, target_dim).
        """

        def repeat(tensor, dim=0):
            return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)

        # blows-up the dimension of each tensor to
        # batch_size * self.num_sample_paths for increasing parallelism
        repeated_past_target_cdf = repeat(past_target_cdf)
        repeated_past_targetT_cdf = repeat(past_target_cdf.sum(2))
        repeated_time_feat = repeat(time_feat)
        repeated_scale = repeat(scale)

        repeated_states1 = [repeat(s, dim=1) for s in begin_states1]
        repeated_states2 = [repeat(s, dim=1) for s in begin_states2]


        future_samples = []

        # for each future time-units we draw new samples for this time-unit
        # and update the state
        for k in range(self.prediction_length):
            
            # (batch_size*num_sample, 1, target_dim, num_lags)
            lags2 = self.get_lagged_subsequences(
                sequence=repeated_past_target_cdf,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            
            # (batch_size*num_sample, 1, 1, num_lags)
            lags1 = lags2.sum(2)
            
            
            rnn_outputs1, repeated_states1, _, _ ,rnn_outputs2, repeated_states2, _, _ = self.unroll(
                begin_state1=repeated_states1,
                begin_state2=repeated_states2,
                scale = repeated_scale,
                lags1=lags1,
                lags2=lags2,
                time_feat=repeated_time_feat[:, k : k + 1, ...],
                unroll_length=1,
            )
            
            
            repeated_scaleT = repeated_scale.sum(2)
            
            rnn_outputs1_permute = rnn_outputs1.permute(0, 2, 1).contiguous().view(rnn_outputs1.shape[0], -1)
            rnn_outputs2_permute = rnn_outputs2.permute(0, 2, 1).contiguous().view(rnn_outputs1.shape[0], -1)
      
            mu = self.softplus(self.distribution_mu(rnn_outputs1_permute))
            sigma = self.softplus(self.distribution_sigma(rnn_outputs1_permute))  + 1e-6
            alpha = torch.squeeze(self.EL(self.distribution_alpha(rnn_outputs2_permute)))
            
            
            #rescaling 
            mu = torch.squeeze(mu * repeated_scaleT)
            sigma = torch.squeeze(sigma*torch.sqrt(repeated_scaleT))

            
            prob = sigma/(sigma+mu)
            batch_size = mu.shape[0]
            #print(sigma.data.cpu().numpy())
            #print(prob.data.cpu().numpy())
            total_count = nbinom.rvs(sigma.data.cpu().numpy(), prob.data.cpu().numpy())  
            total_count[total_count==0] = 1
            if isinstance(total_count, int):
                total_count = np.array([total_count])
            
            
            dirichlet = torch.distributions.dirichlet.Dirichlet(alpha)
            probs = dirichlet.sample()

            
            new_samples = torch.zeros(batch_size,1,alpha.shape[1])

            for b in range(batch_size):
                multinomial = torch.distributions.multinomial.Multinomial(int(total_count[b].item()),probs[b])
                pred = multinomial.sample() #not scaled
                new_samples[b, :, :] = pred 

            # (batch_size, seq_len, target_dim)
            future_samples.append(new_samples)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            repeated_past_target_cdf = torch.cat(
                (repeated_past_target_cdf, new_samples.to(device)), dim=1
            )

        # (batch_size * num_samples, prediction_length, target_dim)
        samples = torch.cat(future_samples, dim=1)

        # (batch_size, num_samples, prediction_length, target_dim)
        return samples.reshape(
            (-1, self.num_parallel_samples, self.prediction_length, self.target_dim,)
        )

    def forward(
        self,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predicts samples given the trained DeepVAR model.
        All tensors should have NTC layout.
        Parameters
        ----------


        past_time_feat
            Dynamic features of past time series (batch_size, history_length,
            num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)

        Returns
        -------
        sample_paths : Tensor
            A tensor containing sampled paths (1, num_sample_paths,
            prediction_length, target_dim).

        """

        # mark padded data as unobserved
        # (batch_size, target_dim, seq_len)
        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        # unroll the decoder in "prediction mode", i.e. with past data only
        _, state1, _, _, _, state2, _, _, scale = self.unroll_encoder(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=None,
            future_target_cdf=None,
        )

        return self.sampling_decoder(
            past_target_cdf=past_target_cdf,
            time_feat=future_time_feat,
            scale = scale,
            begin_states1=state1,
            begin_states2=state2,
        )
