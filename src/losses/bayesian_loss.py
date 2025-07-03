import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def negative_loglikelihood(muA, muP, muN, varA, varP, varN, margin=0.3):
    """
    Compute the negative log-likelihood for Bayesian triplet loss
    """
    dist_ap = torch.sum((muA - muP) ** 2, dim=0)
    dist_an = torch.sum((muA - muN) ** 2, dim=0)
    
    # Add variance terms
    var_ap = torch.sum(varA + varP, dim=0)
    var_an = torch.sum(varA + varN, dim=0)
    
    # Compute log-likelihood
    log_likelihood = -0.5 * (dist_ap / var_ap + torch.log(var_ap) + 
                            dist_an / var_an + torch.log(var_an))
    
    # Add margin term
    loss = torch.clamp(dist_ap - dist_an + margin, min=0.0)
    
    return -log_likelihood + loss


def kl_div_gauss(mu1, var1, mu2, var2):
    """
    Compute KL divergence between two Gaussian distributions
    """
    return 0.5 * (torch.sum(var1 / var2 + (mu1 - mu2) ** 2 / var2 - 1 - torch.log(var1 / var2)))


def kl_div_vMF(mu, var):
    """
    Compute KL divergence for von Mises-Fisher distribution
    """
    # For vMF, we assume mu is normalized and var is the concentration parameter
    kappa = var
    d = mu.size(0)
    
    # Compute Bessel function ratio
    bessel_ratio = torch.sqrt(1 + 4 * kappa ** 2) / (2 * kappa)
    
    # Compute KL divergence
    two_pi = torch.tensor(2 * math.pi, device=kappa.device)
    kl = kappa * bessel_ratio + (d/2 - 1) * torch.log(kappa) - \
         (d/2) * torch.log(two_pi) - torch.log(bessel_ratio)
    
    return kl


class BayesianTripletLoss(nn.Module):
    def __init__(self, margin, varPrior, distribution='gauss'):
        super(BayesianTripletLoss, self).__init__()
        
        self.margin = margin
        self.varPrior = varPrior
        self.kl_scale_factor = 1e-6  # Fixed default value
        self.distribution = distribution

    def forward(self, x, label):
        # divide x into anchor, positive, negative based on labels
        D, N = x.shape
        nq = torch.sum(label.data == -1).item()  # number of tuples
        S = x.size(1) // nq  # number of images per tuple including query: 1+1+n
        A = x[:, label.data == -1].permute(1, 0).repeat(1, S - 2).view((S - 2) * nq, D).permute(1, 0)
        P = x[:, label.data == 1].permute(1, 0).repeat(1, S - 2).view((S - 2) * nq, D).permute(1, 0)
        N = x[:, label.data == 0]

        varA = A[-1:, :]
        varP = P[-1:, :]
        varN = N[-1:, :]

        muA = A[:-1, :]
        muP = P[:-1, :]
        muN = N[:-1, :]

        # calculate nll
        nll = negative_loglikelihood(muA, muP, muN, varA, varP, varN, margin=self.margin)

        kl = torch.tensor(0.0, device=x.device)

        # KL(anchor|| prior) + KL(positive|| prior) + KL(negative|| prior)
        if self.distribution == 'gauss':
            muPrior = torch.zeros_like(muA, requires_grad=False)
            varPrior = torch.ones_like(varA, requires_grad=False) * self.varPrior

            kl = (kl_div_gauss(muA, varA, muPrior, varPrior) + \
                  kl_div_gauss(muP, varP, muPrior, varPrior) + \
                  kl_div_gauss(muN, varN, muPrior, varPrior))

        elif self.distribution == 'vMF':
            kl = (kl_div_vMF(muA, varA) + \
                  kl_div_vMF(muP, varP) + \
                  kl_div_vMF(muN, varN))

        return nll + self.kl_scale_factor * kl

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'
