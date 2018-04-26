from src.feature_model.model import SpectrogramModel as FeatureModel

__all__ = ['FeatureModel']

# TODO: Consider BatchNorm after ReLU
# https://github.com/cvjena/cnn-models/issues/3
# https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout-in-tensorflow
