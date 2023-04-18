"""
Implementation of various mathematical operations in the Poincare ball model of hyperbolic space. Some
functions are based on the implementation in https://github.com/geoopt/geoopt (copyright by Maxim Kochurov).
"""

import numpy as np
import torch
from scipy.special import gamma
import torch.nn as nn

import geoopt
from torch.nn.utils.parametrizations import spectral_norm



def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


# +
class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class RiemannianGradient(torch.autograd.Function):

    c = 1

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # x: B x d

        scale = (1 - RiemannianGradient.c * x.pow(2).sum(-1, keepdim=True)).pow(2) / 4
        return grad_output * scale


# -


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x + torch.sqrt_(1 + x.pow(2))).clamp_min_(1e-5).log_()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


def artanh(x):
    return Artanh.apply(x)


def arsinh(x):
    return Arsinh.apply(x)


def arcosh(x, eps=1e-5):  # pragma: no cover
    x = x.clamp(-1 + eps, 1 - eps)
    return torch.log(x + torch.sqrt(1 + x) * torch.sqrt(x - 1))


def project(x, *, c=1.0):
    r"""
    Safe projection on the manifold for numerical stability. This was mentioned in [1]_
    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    c : float|tensor
        ball negative curvature
    Returns
    -------
    tensor
        projected vector on the manifold
    References
    ----------
    .. [1] Hyperbolic Neural Networks, NIPS2018
        https://arxiv.org/abs/1805.09112
    """
    c = torch.as_tensor(c).type_as(x)
    return _project(x, c)


def _project(x, c):
    norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5)
    maxnorm = (1 - 1e-3) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def lambda_x(x, *, c=1.0, keepdim=False):
    r"""
    Compute the conformal factor :math:`\lambda^c_x` for a point on the ball
    .. math::
        \lambda^c_x = \frac{1}{1 - c \|x\|_2^2}
    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    c : float|tensor
        ball negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    Returns
    -------
    tensor
        conformal factor
    """
    c = torch.as_tensor(c).type_as(x)
    return _lambda_x(x, c, keepdim=keepdim)


def _lambda_x(x, c, keepdim: bool = False):
    return 2 / (1 - c * x.pow(2).sum(-1, keepdim=keepdim))


def mobius_add(x, y, *, c=1.0):
    r"""
    Mobius addition is a special operation in a hyperbolic space.
    .. math::
        x \oplus_c y = \frac{
            (1 + 2 c \langle x, y\rangle + c \|y\|^2_2) x + (1 - c \|x\|_2^2) y
            }{
            1 + 2 c \langle x, y\rangle + c^2 \|x\|^2_2 \|y\|^2_2
        }
    In general this operation is not commutative:
    .. math::
        x \oplus_c y \ne y \oplus_c x
    But in some cases this property holds:
    * zero vector case
    .. math::
        \mathbf{0} \oplus_c x = x \oplus_c \mathbf{0}
    * zero negative curvature case that is same as Euclidean addition
    .. math::
        x \oplus_0 y = y \oplus_0 x
    Another usefull property is so called left-cancellation law:
    .. math::
        (-x) \oplus_c (x \oplus_c y) = y
    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    y : tensor
        point on the Poincare ball
    c : float|tensor
        ball negative curvature
    Returns
    -------
    tensor
        the result of mobius addition
    """
    c = torch.as_tensor(c).type_as(x)
    return _mobius_add(x, y, c)


def _mobius_add(x, y, c):
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / (denom + 1e-5)


def dist(x, y, *, c=1.0, keepdim=False):
    r"""
    Distance on the Poincare ball
    .. math::
        d_c(x, y) = \frac{2}{\sqrt{c}}\tanh^{-1}(\sqrt{c}\|(-x)\oplus_c y\|_2)
    .. plot:: plots/extended/poincare/distance.py
    Parameters
    ----------
    x : tensor
        point on poincare ball
    y : tensor
        point on poincare ball
    c : float|tensor
        ball negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    Returns
    -------
    tensor
        geodesic distance between :math:`x` and :math:`y`
    """
    c = torch.as_tensor(c).type_as(x)
    return _dist(x, y, c, keepdim=keepdim)


def _dist(x, y, c, keepdim: bool = False):
    sqrt_c = c ** 0.5
    dist_c = artanh(sqrt_c * _mobius_add(-x, y, c).norm(dim=-1, p=2, keepdim=keepdim))
    return dist_c * 2 / sqrt_c


def dist0(x, *, c=1.0, keepdim=False):
    r"""
    Distance on the Poincare ball to zero
    Parameters
    ----------
    x : tensor
        point on poincare ball
    c : float|tensor
        ball negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    Returns
    -------
    tensor
        geodesic distance between :math:`x` and :math:`0`
    """
    c = torch.as_tensor(c).type_as(x)
    return _dist0(x, c, keepdim=keepdim)


def _dist0(x, c, keepdim: bool = False):
    sqrt_c = c ** 0.5
    dist_c = artanh(sqrt_c * x.norm(dim=-1, p=2, keepdim=keepdim))
    return dist_c * 2 / sqrt_c


def expmap(x, u, *, c=1.0):
    r"""
    Exponential map for Poincare ball model. This is tightly related with :func:`geodesic`.
    Intuitively Exponential map is a smooth constant travelling from starting point :math:`x` with speed :math:`u`.
    A bit more formally this is travelling along curve :math:`\gamma_{x, u}(t)` such that
    .. math::
        \gamma_{x, u}(0) = x\\
        \dot\gamma_{x, u}(0) = u\\
        \|\dot\gamma_{x, u}(t)\|_{\gamma_{x, u}(t)} = \|u\|_x
    The existence of this curve relies on uniqueness of differential equation solution, that is local.
    For the Poincare ball model the solution is well defined globally and we have.
    .. math::
        \operatorname{Exp}^c_x(u) = \gamma_{x, u}(1) = \\
        x\oplus_c \tanh(\sqrt{c}/2 \|u\|_x) \frac{u}{\sqrt{c}\|u\|_2}
    Parameters
    ----------
    x : tensor
        starting point on poincare ball
    u : tensor
        speed vector on poincare ball
    c : float|tensor
        ball negative curvature
    Returns
    -------
    tensor
        :math:`\gamma_{x, u}(1)` end point
    """
    c = torch.as_tensor(c).type_as(x)
    return _expmap(x, u, c)


def _expmap(x, u, c):  # pragma: no cover
    sqrt_c = c ** 0.5
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
    second_term = (
        tanh(sqrt_c / 2 * _lambda_x(x, c, keepdim=True) * u_norm)
        * u
        / (sqrt_c * u_norm)
    )
    gamma_1 = _mobius_add(x, second_term, c)
    return gamma_1


def expmap0(u, *, c=1.0):
    r"""
    Exponential map for Poincare ball model from :math:`0`.
    .. math::
        \operatorname{Exp}^c_0(u) = \tanh(\sqrt{c}/2 \|u\|_2) \frac{u}{\sqrt{c}\|u\|_2}
    Parameters
    ----------
    u : tensor
        speed vector on poincare ball
    c : float|tensor
        ball negative curvature
    Returns
    -------
    tensor
        :math:`\gamma_{0, u}(1)` end point
    """
    c = torch.as_tensor(c).type_as(u)
    return _expmap0(u, c)


def _expmap0(u, c):
    sqrt_c = c ** 0.5
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return gamma_1


def logmap(x, y, *, c=1.0):
    r"""
    Logarithmic map for two points :math:`x` and :math:`y` on the manifold.
    .. math::
        \operatorname{Log}^c_x(y) = \frac{2}{\sqrt{c}\lambda_x^c} \tanh^{-1}(
            \sqrt{c} \|(-x)\oplus_c y\|_2
        ) * \frac{(-x)\oplus_c y}{\|(-x)\oplus_c y\|_2}
    The result of Logarithmic map is a vector such that
    .. math::
        y = \operatorname{Exp}^c_x(\operatorname{Log}^c_x(y))
    Parameters
    ----------
    x : tensor
        starting point on poincare ball
    y : tensor
        target point on poincare ball
    c : float|tensor
        ball negative curvature
    Returns
    -------
    tensor
        tangent vector that transports :math:`x` to :math:`y`
    """
    c = torch.as_tensor(c).type_as(x)
    return _logmap(x, y, c)


def _logmap(x, y, c):  # pragma: no cover
    sub = _mobius_add(-x, y, c)
    sub_norm = sub.norm(dim=-1, p=2, keepdim=True)
    lam = _lambda_x(x, c, keepdim=True)
    sqrt_c = c ** 0.5
    return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm


def logmap0(y, *, c=1.0):
    r"""
    Logarithmic map for :math:`y` from :math:`0` on the manifold.
    .. math::
        \operatorname{Log}^c_0(y) = \tanh^{-1}(\sqrt{c}\|y\|_2) \frac{y}{\|y\|_2}
    The result is such that
    .. math::
        y = \operatorname{Exp}^c_0(\operatorname{Log}^c_0(y))
    Parameters
    ----------
    y : tensor
        target point on poincare ball
    c : float|tensor
        ball negative curvature
    Returns
    -------
    tensor
        tangent vector that transports :math:`0` to :math:`y`
    """
    c = torch.as_tensor(c).type_as(y)
    return _logmap0(y, c)


def _logmap0(y, c):
    sqrt_c = c ** 0.5
    y_norm = torch.clamp_min(y.norm(dim=-1, p=2, keepdim=True), 1e-5)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)


def mobius_matvec(m, x, *, c=1.0):
    r"""
    Generalization for matrix-vector multiplication to hyperbolic space defined as
    .. math::
        M \otimes_c x = (1/\sqrt{c}) \tanh\left(
            \frac{\|Mx\|_2}{\|x\|_2}\tanh^{-1}(\sqrt{c}\|x\|_2)
        \right)\frac{Mx}{\|Mx\|_2}
    Parameters
    ----------
    m : tensor
        matrix for multiplication
    x : tensor
        point on poincare ball
    c : float|tensor
        negative ball curvature
    Returns
    -------
    tensor
        Mobius matvec result
    """
    c = torch.as_tensor(c).type_as(x)
    return _mobius_matvec(m, x, c)


def _mobius_matvec(m, x, c):
    x_norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5)
    sqrt_c = c ** 0.5
    mx = x @ m.transpose(-1, -2)
    mx_norm = mx.norm(dim=-1, keepdim=True, p=2)
    res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
    cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)
    return _project(res, c)


def _tensor_dot(x, y):
    res = torch.einsum("ij,kj->ik", (x, y))
    return res


def _mobius_addition_batch(x, y, c):
    xy = _tensor_dot(x, y)  # B x C
    x2 = x.pow(2).sum(-1, keepdim=True)  # B x 1
    y2 = y.pow(2).sum(-1, keepdim=True)  # C x 1
    num = 1 + 2 * c * xy + c * y2.permute(1, 0)  # B x C
    num = num.unsqueeze(2) * x.unsqueeze(1)
    num = num + (1 - c * x2).unsqueeze(2) * y  # B x C x D
    denom_part1 = 1 + 2 * c * xy  # B x C
    denom_part2 = c ** 2 * x2 * y2.permute(1, 0)
    denom = denom_part1 + denom_part2
    res = num / (denom.unsqueeze(2) + 1e-5)
    return res


def _hyperbolic_softmax(X, A, P, c):
    lambda_pkc = 2 / (1 - c * P.pow(2).sum(dim=1))
    k = lambda_pkc * torch.norm(A, dim=1) / torch.sqrt(c)
    mob_add = _mobius_addition_batch(-P, X, c)
    num = 2 * torch.sqrt(c) * torch.sum(mob_add * A.unsqueeze(1), dim=-1)
    denom = torch.norm(A, dim=1, keepdim=True) * (1 - c * mob_add.pow(2).sum(dim=2))
    logit = k.unsqueeze(1) * arsinh(num / denom)
    return logit.permute(1, 0)


def p2k(x, c):
    denom = 1 + c * x.pow(2).sum(-1, keepdim=True)
    return 2 * x / denom


def k2p(x, c):
    denom = 1 + torch.sqrt(1 - c * x.pow(2).sum(-1, keepdim=True))
    return x / denom


def lorenz_factor(x, *, c=1.0, dim=-1, keepdim=False):
    """

    Parameters
    ----------
    x : tensor
        point on Klein disk
    c : float
        negative curvature
    dim : int
        dimension to calculate Lorenz factor
    keepdim : bool
        retain the last dim? (default: false)

    Returns
    -------
    tensor
        Lorenz factor
    """
    return 1 / torch.sqrt(1 - c * x.pow(2).sum(dim=dim, keepdim=keepdim))


def poincare_mean(x, dim=0, c=1.0):
    x = p2k(x, c)
    lamb = lorenz_factor(x, c=c, keepdim=True)
    mean = torch.sum(lamb * x, dim=dim, keepdim=True) / torch.sum(
        lamb, dim=dim, keepdim=True
    )
    mean = k2p(mean, c)
    return mean.squeeze(dim)


def _dist_matrix(x, y, c):
    sqrt_c = c ** 0.5
    return (
        2
        / sqrt_c
        * artanh(sqrt_c * torch.norm(_mobius_addition_batch(-x, y, c=c), dim=-1))
    )


def dist_matrix(x, y, c=1.0):
    c = torch.as_tensor(c).type_as(x)
    return _dist_matrix(x, y, c)


def auto_select_c(d):
    """
    calculates the radius of the Poincare ball,
    such that the d-dimensional ball has constant volume equal to pi
    """
    dim2 = d / 2.0
    R = gamma(dim2 + 1) / (np.pi ** (dim2 - 1))
    R = R ** (1 / float(d))
    c = 1 / (R ** 2)
    return c

"""
    Taken from https://github.com/twitter-research/hyperbolic-rl
"""

class PoincareDist:
    def __init__(self, c=1.0, euclidean_inputs=True):
        self.euclidean_inputs = True
        self.hyper_ball = geoopt.PoincareBall(c=1)
    
    def dist2(self, x, y):
        if self.euclidean_inputs:
            x, y = self.hyper_ball.expmap0(x), self.hyper_ball.expmap0(y)
        return self.hyper_ball.dist2(x, y)

# class PoincareDist:
#     def __init__(self, c=1.0, project_input=True, euclidean_inputs=True):
#         self.project_input = project_input
#         self.euclidean_inputs = euclidean_inputs
#         self.ball = geoopt.PoincareBall(c)
    
#     def map_to_ball(self, input):
#         return self.ball.expmap0(input, project=self.project_input)

#     def manual_distance(self, points, other_points):
#         dist = torch.arccosh(1 + 2 * (points - other_points).pow(2).sum(-1) / (1 - points.pow(2).sum(-1)) / (
#                     1 - other_points.pow(2).sum(-1)))
#         return dist

#     def distance(self, x, y):
#         if self.euclidean_inputs:
#             x = self.map_to_ball(x)
#             y = self.map_to_ball(y)
#         return self.manual_distance(x, y)

#     def distance_matrix(self, input):
#         if self.euclidean_inputs:
#             input = self.map_to_ball(input)
#         input = input[:, None, :]
#         distances = self.manual_distance(input.unsqueeze(0), input.unsqueeze(1))
#         return distances.sum(-1)


class PoincarePlaneDistance(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            num_planes: int,  # out_features
            c=1.0,
            euclidean_inputs=True,
            rescale_euclidean_norms_gain=None,  # rescale euclidean norms based on the dimensions per space
            signed=True,
            scaled=True,
            squared=False,
            project_input=True,
            normal_std=None,
            dimensions_per_space=None,
            rescale_normal_params=False,
            effective_softmax_rescale=None,
            hyperbolic_representation_metric=None,
    ):
        super().__init__()
        self.euclidean_inputs = euclidean_inputs
        self.rescale_norms_gain = rescale_euclidean_norms_gain
        self.signed = signed
        self.scaled = scaled
        self.squared = squared
        self.project_input = project_input
        self.ball = geoopt.PoincareBall(c=c)
        self.in_features = in_features
        self.num_planes = num_planes
        self.rescale_normal_params = rescale_normal_params

        if effective_softmax_rescale is not None:
            if self.rescale_normal_params:
                self.logits_multiplier = effective_softmax_rescale
            else:
                self.logits_multiplier = effective_softmax_rescale * 2
        else:
            self.logits_multiplier = 1

        if dimensions_per_space is not None:
            assert in_features % dimensions_per_space == 0
            self.dimensions_per_space = dimensions_per_space
            self.num_spaces = in_features // dimensions_per_space
        else:
            self.dimensions_per_space = self.in_features
            self.num_spaces = 1

        self.normals = nn.Parameter(torch.empty((num_planes, self.num_spaces, self.dimensions_per_space)))
        self.bias = geoopt.ManifoldParameter(torch.zeros(num_planes, self.num_spaces, self.dimensions_per_space),
                                             manifold=self.ball)

        self.normal_std = normal_std
        self.reset_parameters()

        self.hyperbolic_representation_metric = hyperbolic_representation_metric
        if self.hyperbolic_representation_metric is not None and self.euclidean_inputs:
            self.hyperbolic_representation_metric.add('hyperbolic_representations')

    def get_mean_norm(self, input):
        if self.dimensions_per_space:
            input_shape = input.size()
            input_batch_dims = input_shape[:-1]
            input_feature_dim = input_shape[-1]
            rs_input = input.view(*input_batch_dims, input_feature_dim // self.dimensions_per_space,
                                  self.dimensions_per_space)
        else:
            rs_input = input
        return torch.norm(rs_input, p=2, dim=-1, keepdim=True).mean()

    def map_to_ball(self, input):  # input bs x in_feat
        if self.rescale_norms_gain:  # make expected tangent vector norm independent of initial dimension (approximately)
            input = self.rescale_norms_gain * input / np.sqrt(self.dimensions_per_space)
        return self.ball.expmap0(input, project=self.project_input)

    def manual_distance(self, points, other_points):
        dist = torch.arccosh(1 + 2 * (points - other_points).pow(2).sum(-1) / (1 - points.pow(2).sum(-1)) / (
                    1 - other_points.pow(2).sum(-1)))
        return dist

    def distance_matrix(self, input, euclidean_inputs=True, cpu=False):
        if euclidean_inputs:
            input = self.map_to_ball(input)
        input_batch_dims = input.size()[:-1]
        input = input.view(*input_batch_dims, self.num_spaces, self.dimensions_per_space)
        if cpu:
            input = input.cpu()
        distances = self.manual_distance(input.unsqueeze(0), input.unsqueeze(1))

        return distances.sum(-1)

    def distance_to_space(self, input, other, euclidean_inputs):
        if euclidean_inputs:
            input = self.map_to_ball(input)
            other = self.map_to_ball(other)
        input_batch_dims = input.size()[:-1]
        input = input.view(-1, self.num_spaces, self.dimensions_per_space)
        other = other.view(-1, self.num_spaces, self.dimensions_per_space)
        summed_dists = self.ball.dist(x=input, y=other).sum(-1)
        return summed_dists.view(input_batch_dims)

    def forward(self, input):  # input bs x in_feat
        input_batch_dims = input.size()[:-1]
        input = input.view(-1, self.num_spaces, self.dimensions_per_space)
        if self.euclidean_inputs:
            input = self.map_to_ball(input)
            if self.hyperbolic_representation_metric is not None:
                self.hyperbolic_representation_metric.set(hyperbolic_representations=input)
        input_p = input.unsqueeze(-3)  # bs x 1 x num_spaces x dim_per_space
        if self.rescale_normal_params:
            conformal_factor = 1 - self.bias.pow(2).sum(dim=-1)
            a = self.normals * conformal_factor.unsqueeze(-1)
        else:
            a = self.normals
        distances = self.ball.dist2plane(x=input_p, p=self.bias, a=a,
                                         signed=self.signed, scaled=self.scaled, dim=-1)
        if self.rescale_normal_params:
            distances = distances * 2 / conformal_factor
        distance = distances.sum(-1)
        distance = distance.view(*input_batch_dims, self.num_planes)
        return distance * self.logits_multiplier

    def forward_rs(self, input):  # input bs x in_feat
        input_batch_dims = input.size()[:-1]
        input = input.view(-1, self.num_spaces, self.dimensions_per_space)
        if self.euclidean_inputs:
            input = self.map_to_ball(input)
            if self.hyperbolic_representation_metric is not None:
                self.hyperbolic_representation_metric.set(hyperbolic_representations=input)
        input_p = input.unsqueeze(-3)  # bs x 1 x num_spaces x dim_per_space
        conformal_factor = 1 - self.bias.pow(2).sum(dim=-1)
        distances = self.ball.dist2plane(x=input_p, p=self.bias, a=self.normals * conformal_factor.unsqueeze(-1),
                                         signed=self.signed, scaled=self.scaled, dim=-1)
        distances = distances * 2 / conformal_factor
        distance = distances.sum(-1)
        distance = distance.view(*input_batch_dims, self.num_planes)
        return distance

    def extra_repr(self):
        return (
            "poincare_dim={num_spaces}x{dimensions_per_space} ({in_features}), "
            "num_planes={num_planes}, "
            .format(**self.__dict__))

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.zeros_(self.bias)
        if self.normal_std:
            nn.init.normal_(self.normals, std=self.normal_std)
        else:
            nn.init.normal_(self.normals, std=1 / np.sqrt(self.in_features))

def weight_init_hyp(m):
    if isinstance(m, PoincarePlaneDistance):
        nn.init.normal_(m.normals.data, 1 / np.sqrt(m.in_features))
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    else:
        weight_init(m)

def final_weight_init_hyp(m):
    if isinstance(m, PoincarePlaneDistance):
        nn.init.normal_(m.normals.data, 1 / np.sqrt(m.in_features))
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Linear):
        final_weight_init(m=m)

def final_weight_init_hyp_small(m):
    if isinstance(m, PoincarePlaneDistance):
        nn.init.normal_(m.normals.data, 1 / np.sqrt(m.in_features) * 0.01)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Linear):
        final_weight_init(m=m)

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

def final_weight_init(m):
    nn.init.orthogonal_(m.weight.data, gain=0.01)
    if hasattr(m.bias, 'data'):
        m.bias.data.fill_(0.0)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ClipNorm(nn.Module):
    def __init__(self, max_norm=15, dimensions_per_space=None):
        super().__init__()
        self.max_norm = max_norm
        self.dimension_per_space = dimensions_per_space

    def get_mean_norm(self, input):
        if self.dimension_per_space:
            input_shape = input.size()
            input_batch_dims = input_shape[:-1]
            input_feature_dim = input_shape[-1]
            rs_input = input.view(*input_batch_dims, input_feature_dim // self.dimension_per_space,
                                  self.dimension_per_space)
        else:
            rs_input = input
        return torch.norm(rs_input, p=2, dim=-1, keepdim=True).mean()

    def forward(self, input):  # input bs x in_feat
        if self.dimension_per_space:
            input_shape = input.size()
            input_batch_dims = input_shape[:-1]
            input_feature_dim = input_shape[-1]
            rs_input = input.view(*input_batch_dims, input_feature_dim // self.dimension_per_space,
                                  self.dimension_per_space)
        else:
            rs_input = input
        input_l2 = torch.norm(rs_input, p=2, dim=-1, keepdim=True)
        clipped_input = torch.minimum(self.max_norm / input_l2,
                                      torch.ones_like(input_l2)) * rs_input
        if self.dimension_per_space:
            clipped_input = clipped_input.view(*input_shape)
        return clipped_input

def apply_sn_until_instance(modules, layer_instance):
    reached_instance = False
    application_modules = []
    for module in modules:
        if isinstance(module, layer_instance):
            reached_instance = True
        elif not reached_instance:
            application_modules.append(module)
    for module in application_modules:
        module.apply(apply_sn)

def apply_sn(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        return spectral_norm(m)
    else:
        return m