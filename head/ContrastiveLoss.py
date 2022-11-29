from torch.nn import Module
from pytorch_metric_learning import losses, distances

#  ContrastiveLoss interface for module https://kevinmusgrave.github.io/pytorch-metric-learning/losses/

class ContrastiveLoss(Module):


    def __init__(self,  distance=None):
        super(ContrastiveLoss, self).__init__()

        if distance == "cosine":
            self.distance = distances.CosineSimilarity()
            self.pos_margin = 1 #recommended with cosine similarity
            self.neg_margin = 0
        else:
            self.distance = distances.LpDistance(normalize_embeddings=True, p=2, power=1)
            self.pos_margin = 0
            self.neg_margin = 1
        self.loss_func = losses.ContrastiveLoss(pos_margin=self.pos_margin, neg_margin=self.neg_margin, distance=self.distance)
    def forward(self, feats, labels, ref_emb=None, ref_labels=None):
        loss = self.loss_func(feats, labels, ref_emb=ref_emb, ref_labels=ref_labels)
        return loss
