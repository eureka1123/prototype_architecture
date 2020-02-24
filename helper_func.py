import torch

def list_of_distances(X,Y):
    XX = list_of_norms(X)
    XX = XX.view(-1,1)
    YY = list_of_norms(Y)
    YY = YY.view(1,-1)
    output = XX + YY - 2*torch.matmul(X,Y.transpose(0,1))
    return output

def list_of_norms(X):
    x = torch.pow(X,2)
    x = x.view(x.shape[0],-1)
    x = x.sum(1)
    return x