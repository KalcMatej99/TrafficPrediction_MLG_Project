import torch

def z_score(x, mean, std):
    return (x - mean) / std
def un_z_score(x_normed, mean, std):
    return x_normed * std  + mean
def MAPE(v, v_):
    return torch.mean(torch.abs((v_ - v)) /(v + 1e-15) * 100)
def RMSE(v, v_):
    return torch.sqrt(torch.mean((v_ - v) ** 2))
def MAE(v, v_):
    return torch.mean(torch.abs(v_ - v))