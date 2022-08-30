from typing import List, Optional, Callable

import numpy as np
import pandas as pd
import torch

from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import TimeFeature
from gluonts.torch.modules.distribution_output import DistributionOutput
from gluonts.torch.util import copy_parameters
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.model.predictor import Predictor
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    CDFtoGaussianTransform,
    Chain,
    ExpandDimArray,
    ExpectedNumInstanceSampler,
    InstanceSampler,
    InstanceSplitter,
    RenameFields,
    SelectFields,
    SetFieldIfNotPresent,
    TargetDimIndicator,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
    cdf_to_gaussian_forward_transform,
)




from pts import Trainer
from pts.model.utils import get_module_forward_input_names
from pts.feature import (
    fourier_time_features_from_frequency,
    lags_for_fourier_time_features_from_frequency,
)
from pts.model import PyTorchEstimator

from ._network import DeepNEGPOLTrainingNetwork, DeepNEGPOLPredictionNetwork

from pandas.tseries.frequencies import to_offset
from gluonts.time_feature import TimeFeature, norm_freq_str






class FourierDateFeatures(TimeFeature):
    @validated()
    def __init__(self, freq: str) -> None:
        super().__init__()
        # reocurring freq
        freqs = [
            "month",
            "day",
            "hour",
            "minute",
            "weekofyear",
            "weekday",
            "dayofweek",
            "dayofyear",
            "daysinmonth",
        ]

        assert freq in freqs
        self.freq = freq

    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        values = getattr(index, self.freq)
        num_values = max(values) + 1
        steps = [x * 2.0 * np.pi / num_values for x in values]
        return np.vstack([np.cos(steps), np.sin(steps)])


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    features = {
        "M": ["weekofyear"],
        "W": ["daysinmonth", "weekofyear"],
        "D": ["dayofweek"],
        "B": ["dayofweek", "dayofyear"],
        "H": ["hour", "dayofweek"],
        "min": ["minute", "hour", "dayofweek"],
        "T": ["minute", "hour", "dayofweek"],
    }

    offset = to_offset(freq_str)
    granularity = norm_freq_str(offset.name)
    assert granularity in features, f"freq {granularity} not supported"

    feature_classes: List[TimeFeature] = [
        FourierDateFeatures(freq=freq) for freq in features[granularity]
    ]
    return feature_classes


def get_lags_for_frequency(
    freq_str: str, num_lags: Optional[int] = None
) -> List[int]:
    offset = to_offset(freq_str)

    if offset.name == "M":
        lags = [[1, 12]]
    elif offset.name == "D":
        lags = [[1, 7, 14]]
    elif offset.name == "B":
        lags = [[1, 2]]
    elif offset.name == "H":
        lags = [[1, 24, 168]]
    elif offset.name in ("min", "T"):
        lags = [[1, 4, 12, 24, 48]]
    else:
        lags = [[1]]

    # use less lags
    output_lags = list(int(lag) for sub_list in lags for lag in sub_list)
    output_lags = sorted(list(set(output_lags)))
    return output_lags[:num_lags]





class DeepNEGPOLEstimator(PyTorchEstimator):
    def __init__(
        self,
        input_size1: int,
        input_size2: int,
        freq: str,
        prediction_length: int,
        target_dim: int,
        trainer: Trainer = Trainer(),
        num_layers1: int = 2,
        num_layers2: int = 2,
        num_cells1: int = 40,
        num_cells2: int = 40,
        context_length: Optional[int] = None,
        num_parallel_samples: int = 100,
        dropout_rate: float = 0.1,
        use_feat_dynamic_real: bool = False,
        use_feat_static_cat: bool = False,
        use_feat_static_real: bool = False,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        rank: Optional[int] = 5,
        scaling: bool = True,
        pick_incomplete: bool = False,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        conditioning_length: int = 200,
        **kwargs,
    ) -> None:
        super().__init__(trainer=trainer, **kwargs)

        self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )



        self.input_size1 = input_size1
        self.input_size2 = input_size2
        self.num_layers1 = num_layers1
        self.num_layers2 = num_layers2
        self.num_cells1 = num_cells1
        self.num_cells2 = num_cells2
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.num_parallel_samples = num_parallel_samples
        self.dropout_rate = dropout_rate
        self.use_feat_dynamic_real = use_feat_dynamic_real
        self.use_feat_static_cat = use_feat_static_cat
        self.use_feat_static_real = use_feat_static_real
        self.cardinality = cardinality if cardinality and use_feat_static_cat else [1]
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None
            else [min(50, (cat + 1) // 2) for cat in self.cardinality]
        )
        self.conditioning_length = conditioning_length


        #self.time_features = None

        self.pick_incomplete = pick_incomplete
        self.scaling = scaling

        self.output_transform = None
        



        self.lags_seq = (
            lags_seq
            if lags_seq is not None
            else get_lags_for_frequency(freq_str=freq)
        )

        self.time_features =  time_features_from_frequency_str(freq)
        

        self.history_length = self.context_length + max(self.lags_seq)



        self.train_sampler =  ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_past=0 if pick_incomplete else self.history_length,
                min_future=prediction_length,
            )
        
        self.validation_sampler =  ValidationSplitSampler(
                min_past=0 if pick_incomplete else self.history_length,
                min_future=prediction_length,
            )
        

    def create_transformation(self) -> Transformation:
        return Chain(
            [
               # AsNumpyArray(
               #     field=FieldName.TARGET,
               #     expected_ndim=1 + len(self.distr_output.event_shape),
               # ),
                # maps the target to (1, T)
                # if the target data is uni dimensional
                #ExpandDimArray(
                #    field=FieldName.TARGET,
                #    axis=0 if self.distr_output.event_shape[0] == 1 else None,
                #),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME],
                ),
                SetFieldIfNotPresent(
                    field=FieldName.FEAT_STATIC_CAT, value=[0.0]
                ),
                TargetDimIndicator(
                    field_name="target_dimension_indicator",
                    target_field=FieldName.TARGET,
                ),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
            ]
        )

    def create_instance_splitter(self, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.history_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES
            ],
        ) + (
            RenameFields(
                {
                    f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_cdf",
                    f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_cdf",
                }
            )
        )

    def create_training_network(self, device: torch.device) -> DeepNEGPOLTrainingNetwork:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DeepNEGPOLTrainingNetwork(
            input_size1=self.input_size1,
            input_size2=self.input_size2,
            target_dim=self.target_dim,
            num_layers1=self.num_layers1,
            num_layers2=self.num_layers2,
            num_cells1=self.num_cells1,
            num_cells2=self.num_cells2,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
        ).to(device)

    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: DeepNEGPOLTrainingNetwork,
        device: torch.device,
    ) -> Predictor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prediction_network = DeepNEGPOLPredictionNetwork(
            input_size1=self.input_size1,
            input_size2=self.input_size2,
            target_dim=self.target_dim,
            num_parallel_samples=self.num_parallel_samples,
            num_layers1=self.num_layers1,
            num_layers2=self.num_layers2,
            num_cells1=self.num_cells1,
            num_cells2=self.num_cells2,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
        ).to(device)

        copy_parameters(trained_network, prediction_network)
        input_names = get_module_forward_input_names(prediction_network)
        prediction_splitter = self.create_instance_splitter("test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=input_names,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            device=device,
            output_transform=self.output_transform,
        )
