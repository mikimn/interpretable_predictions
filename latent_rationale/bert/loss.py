import torch
from torch import nn

from ..common.util import get_z_stats


# noinspection DuplicatedCode
class RationaleLoss(nn.Module):
    def __init__(self,
                 selection: float = 1.,
                 lasso: float = 0.,
                 lagrange_alpha: float = 0.5,
                 lagrange_lr: float = 0.05,
                 # lambda_init: float = 0.0015,
                 lambda_init: float = 0.01,
                 lambda_min: float = 1e-12,
                 lambda_max: float = 5.,
                 lambda_sparsity: float = 0.1,
                 lambda_lasso: float = 0.5
                 ):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.selection = selection

        self.selection = selection
        self.lasso = lasso

        self.alpha = lagrange_alpha
        self.lagrange_lr = lagrange_lr
        self.lambda_init = lambda_init
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        self.lambda_sparsity = lambda_sparsity
        self.lambda_lasso = lambda_lasso

        # lagrange buffers
        self.register_buffer('lambda0', torch.full((1,), lambda_init))
        self.register_buffer('lambda1', torch.full((1,), lambda_init))
        self.register_buffer('c0_ma', torch.full((1,), 0.))  # moving average
        self.register_buffer('c1_ma', torch.full((1,), 0.))  # moving average

    @staticmethod
    def _l0_regularize(z_dists, mask):
        # L0 regularizer (sparsity constraint)
        # pre-compute for regularizers: pdf(0.)
        mask = mask.byte()
        batch_size = mask.size(0)
        lengths = mask.sum(1).float()  # [B]
        if len(z_dists) == 1:
            pdf0 = z_dists[0].pdf(0.)
        else:
            pdf0 = []
            for t in range(len(z_dists)):
                pdf_t = z_dists[t].pdf(0.)
                pdf0.append(pdf_t)
            pdf0 = torch.stack(pdf0, dim=1)  # [B, T, 1]

        pdf0 = pdf0.squeeze(-1)
        pdf0 = torch.where(mask, pdf0, pdf0.new_zeros([1]))  # [B, T]

        pdf_nonzero = 1. - pdf0  # [B, T]
        pdf_nonzero = torch.where(mask, pdf_nonzero, pdf_nonzero.new_zeros([1]))

        l0 = pdf_nonzero.sum(1) / (lengths + 1e-9)  # [B]
        l0 = l0.sum() / batch_size
        return l0, pdf0, pdf_nonzero

    def _l0_relax(self, l0):
        c0_hat = (l0 - self.selection)

        # moving average of the constraint
        self.c0_ma = self.alpha * self.c0_ma + (1 - self.alpha) * c0_hat.item()

        # compute smoothed constraint (equals moving average c0_ma)
        c0 = c0_hat + (self.c0_ma.detach() - c0_hat.detach())

        # update lambda
        self.lambda0 = self.lambda0 * torch.exp(self.lagrange_lr * c0.detach())
        self.lambda0 = self.lambda0.clamp(self.lambda_min, self.lambda_max)
        return c0_hat, c0

    @staticmethod
    def _l1_regularize(pdf0, pdf_nonzero, mask):
        # fused lasso (coherence constraint)
        mask = mask.byte()
        batch_size = mask.size(0)
        lengths = mask.sum(1).float()  # [B]

        # cost z_t = 0, z_{t+1} = non-zero
        zt_zero = pdf0[:, :-1]
        ztp1_nonzero = pdf_nonzero[:, 1:]

        # cost z_t = non-zero, z_{t+1} = zero
        zt_nonzero = pdf_nonzero[:, :-1]
        ztp1_zero = pdf0[:, 1:]

        # number of transitions per sentence normalized by length
        lasso_cost = zt_zero * ztp1_nonzero + zt_nonzero * ztp1_zero
        lasso_cost = lasso_cost * mask.float()[:, :-1]
        lasso_cost = lasso_cost.sum(1) / (lengths + 1e-9)  # [B]
        lasso_cost = lasso_cost.sum() / batch_size
        return lasso_cost

    def _l1_relax(self, l1):
        lasso_cost = l1
        # lagrange coherence dissatisfaction (batch average)
        target1 = self.lasso

        # lagrange dissatisfaction, batch average of the constraint
        c1_hat = (lasso_cost - target1)

        # update moving average
        self.c1_ma = self.alpha * self.c1_ma + (1 - self.alpha) * c1_hat.detach()

        # compute smoothed constraint
        c1 = c1_hat + (self.c1_ma.detach() - c1_hat.detach())

        # update lambda
        self.lambda1 = self.lambda1 * torch.exp(self.lagrange_lr * c1.detach())
        self.lambda1 = self.lambda1.clamp(self.lambda_min, self.lambda_max)
        return c1_hat, c1

    def forward(self, preds, targets, z_dists, mask):
        # preds: [N, C]
        # targets: [N,]
        # mask: [B, S]
        loss = self.criterion(preds, targets)  # [B, T]
        optional = {
            "ce": loss
        }
        if z_dists is None:
            return loss, optional

        # L0 regularizer (sparsity constraint)
        # pre-compute for regularizers: pdf(0.)
        l0, pdf0, pdf_nonzero = self._l0_regularize(z_dists, mask)
        c0_hat, c0 = self._l0_relax(l0)
        # c0 = l0

        with torch.no_grad():
            optional["cost0_l0"] = l0.item()
            optional["target0"] = self.selection
            optional["c0_hat"] = c0_hat.item()
            optional["c0"] = c0.item()  # same as moving average
            optional["lambda0"] = self.lambda0.item()
            optional["lagrangian0"] = (self.lambda0 * c0_hat).item()
            optional["a"] = z_dists[0].a.mean().item()
            optional["b"] = z_dists[0].b.mean().item()

        lambda0 = self.lambda0.squeeze().detach()
        # print(loss)
        # print(lambda0)
        # print(c0)
        # FIXME
        # loss += lambda0 * c0.squeeze()
        loss += self.lambda_sparsity * l0.squeeze()

        # number of transitions per sentence normalized by length
        # TODO Number of transitions per premise/hypothesis??
        lasso_cost = self._l1_regularize(pdf0, pdf_nonzero, mask)
        c1_hat, c1 = self._l1_relax(lasso_cost)
        # c1 = lasso_cost

        with torch.no_grad():
            optional["cost1_lasso"] = lasso_cost.item()
            optional["target1"] = self.lasso
            optional["c1_hat"] = c1_hat.item()
            optional["c1"] = c1.item()  # same as moving average
            optional["lambda1"] = self.lambda1.item()
            optional["lagrangian1"] = (self.lambda1 * c1_hat).item()

        # lambda1 = self.lambda1.squeeze().detach()
        # FIXME
        # loss += self.lambda_sparsity * c1.squeeze()
        loss += self.lambda_lasso * lasso_cost.squeeze()
        # loss += LAMBDA_TEST * c1.squeeze()
        optional["no_ce"] = loss - optional["ce"]
        return loss, optional
