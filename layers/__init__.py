from layers.loss.center_loss import CenterLoss
from layers.loss.triplet_loss import TripletLoss
from layers.loss.adaptive_triplet import AdaptiveTripletLoss
from layers.loss.multi_observer import MultiObserverLoss
from layers.loss.center_cluster import CenterClusterLoss
from layers.loss.weighted_pair_loss import WeightedPairLoss
from layers.loss.CrossEntropyLabelSmooth import CrossEntropyLabelSmooth
from layers.module.norm_linear import NormalizeLinear
from layers.loss.AngleAlignment import AngleAlignmentLoss
from layers.loss.MMDLoss import MMD_loss
__all__ = ['CenterLoss', 
           'TripletLoss', 
           'AdaptiveTripletLoss',
           'AngleAlignmentLoss',
           'MultiObserverLoss',
           'CenterClusterLoss',
           'WeightedPairLoss',
           'CrossEntropyLabelSmooth',
           'NormalizeLinear']