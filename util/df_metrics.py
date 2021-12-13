from torchmetrics.metric import Metric
import torch
from external.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D


class IoU(Metric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("iou_sum", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0).float(), dist_reduce_fx="sum")

    # noinspection PyMethodOverriding
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        intersection = (preds & target).sum(-1).sum(-1).sum(-1).squeeze(1)
        union = (preds | target).sum(-1).sum(-1).sum(-1).squeeze(1)
        valid_mask = union > 0
        intersection = intersection[valid_mask]
        union = union[valid_mask]
        if union.sum() > 0:
            self.iou_sum += (intersection / (union + 1e-5)).sum()
            self.total += intersection.shape[0]

    def compute(self):
        return self.iou_sum.float() / self.total


class Chamfer3D(Metric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cham_loss = dist_chamfer_3D.chamfer_3DDist()
        self.add_state("cd_sum", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0).float(), dist_reduce_fx="sum")

    # noinspection PyMethodOverriding
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        cd = torch.tensor(0).float().to(device=preds.device)
        preds = preds.squeeze(1)
        target = target.squeeze(1)
        valid_chamf = 0
        for ip in range(preds.shape[0]):
            points_pred = torch.nonzero(preds[ip], as_tuple=False).unsqueeze(0).float()
            points_target = torch.nonzero(target[ip], as_tuple=False).unsqueeze(0).float()
            if points_pred.shape[0] > 0 and points_target.shape[0] > 0:
                dist1, dist2, _, _ = self.cham_loss(points_target, points_pred)
                cd_ = (torch.mean(dist1)) + (torch.mean(dist2))
                if not torch.isnan(cd_):
                    cd += cd_
                    valid_chamf += 1
        self.cd_sum += cd
        self.total += valid_chamf

    def compute(self):
        return self.cd_sum.float() / self.total


class Precision(Metric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("precision_sum", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0).float(), dist_reduce_fx="sum")

    # noinspection PyMethodOverriding
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        intersection = (preds & target).sum(-1).sum(-1).sum(-1).squeeze(1)
        self.precision_sum += (intersection / (preds.sum(-1).sum(-1).sum(-1).squeeze(1) + 1e-5)).sum()
        self.total += intersection.shape[0]

    def compute(self):
        return self.precision_sum.float() / self.total


class Recall(Metric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("recall_sum", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0).float(), dist_reduce_fx="sum")

    # noinspection PyMethodOverriding
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        intersection = (preds & target).sum(-1).sum(-1).sum(-1).squeeze(1)
        self.recall_sum += (intersection / (target.sum(-1).sum(-1).sum(-1).squeeze(1) + 1e-5)).sum()
        self.total += intersection.shape[0]

    def compute(self):
        return self.recall_sum.float() / self.total
