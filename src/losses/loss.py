from losses.bayesian_loss import BayesianTripletLoss
from pytorch_metric_learning import losses, distances


def configure_metric_loss(loss, distance, margin, varPrior=1.0, distribution='gauss'):

    if distance == "cosine":
        dist = distances.CosineSimilarity()
    elif distance == "euclidean":
        dist = distances.LpDistance()
    else:
        raise ValueError(f"Unsupported distance: {distance}")

    if loss == "triplet":
        criterion = losses.TripletMarginLoss(margin=margin, distance=dist)
    elif loss in ("contrastive", "arccos"):
        pos_margin = margin if distance == "dot" else 0
        neg_margin = 0 if distance == "dot" else margin

        criterion = losses.ContrastiveLoss(
            pos_margin=pos_margin, neg_margin=neg_margin, distance=dist
        )

    elif loss == "bayesian":
        criterion = BayesianTripletLoss(margin=margin, varPrior=varPrior, distribution=distribution)

    else:
        raise NotImplementedError(f"Unsupported loss: {loss}")

    return criterion
