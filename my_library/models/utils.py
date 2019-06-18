import torch


def get_moments(dist: torch.Tensor) -> torch.Tensor:
    """
    Returns the first 4 moments of the input data.
    We'll (potentially) use this as the input to our discriminator.
    """
    mean = torch.mean(dist)
    diffs = dist - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0))
    kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0  # excess kurtosis, should be 0 for Gaussian
    final = torch.cat((mean.reshape(1,), std.reshape(1,), skews.reshape(1,), kurtoses.reshape(1,)))
    return final