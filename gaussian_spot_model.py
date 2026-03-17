# Adapted from tapqir/distributions/util.py
import math
import torch


def gaussian_spots(
    height: torch.Tensor,  # (N, F, C, K) or (N, F, Q, 1, K)
    width: torch.Tensor,  # (N, F, C, K) or (N, F, Q, 1, K)
    x: torch.Tensor,  # (N, F, C, K) or (N, F, Q, 1, K)
    y: torch.Tensor,  # (N, F, C, K) or (N, F, Q, 1, K)
    target_locs: torch.Tensor,  # (N, F, C, 1, 2) or (N, F, 1, C, 1, 2)
    P: int,
    m: torch.Tensor = None,
) -> torch.Tensor:
    r"""
    Calculates ideal shape of the 2D-Gaussian spots given spot parameters
    and target positions.

    .. math::
        \mu^S_{\mathsf{pixelX}(i), \mathsf{pixelY}(j)} =
        \dfrac{m \cdot h}{2 \pi w^2}
        \exp{\left( -\dfrac{(i-x-x^\mathsf{target})^2 + (j-y-y^\mathsf{target})^2}{2 w^2} \right)}

    :param height: Integrated spot intensity. Should be broadcastable to ``batch_shape``.
    :param width: Spot width. Should be broadcastable to ``batch_shape``.
    :param x: Spot center on x-axis. Should be broadcastable to ``batch_shape``.
    :param y: Spot center on y-axis. Should be broadcastable to ``batch_shape``.
    :param target_locs: Target location. Should have
        the rightmost size ``2`` corresponding to locations on
        x- and y-axes, and be broadcastable to ``batch_shape + (2,)``.
    :param P: Number of pixels along the axis.
    :param m: Spot presence indicator. Should be broadcastable to ``batch_shape``.
    :return: A tensor of a shape ``batch_shape + (P, P)`` representing 2D-Gaussian spots.
    """
    # create meshgrid of PxP pixel positions
    device = height.device
    P_range = torch.arange(P, device=device)
    i_pixel, j_pixel = torch.meshgrid(P_range, P_range, indexing="xy")
    ij_pixel = torch.stack((i_pixel, j_pixel), dim=-1)

    # Ideal 2D gaussian spots
    spot_locs = target_locs + torch.stack((x, y), -1)
    scale = width[..., None, None, None]
    loc = spot_locs[..., None, None, :]
    var = scale**2
    normalized_gaussian = torch.exp(
    (
        -((ij_pixel - loc) ** 2) / (2 * var)
        - 2 * scale.log()  # Correct for 2D: 2 * log(width)
        - math.log(2 * math.pi)  # Correct for 2D: log(2*pi)
    ).sum(-1)
    )  # (N, F, C, K, P, P) or (N, F, Q, C, K, P, P)
    if m is not None:
        height = m * height
    return height[..., None, None] * normalized_gaussian


def generate_gaussian_spot(height, width, x, y, target_x, target_y, P):
    """
    Simple wrapper to generate a single 2D Gaussian spot.

    :param height: Spot intensity (scalar)
    :param width: Spot width (scalar)
    :param x: Spot center x relative to target
    :param y: Spot center y relative to target
    :param target_x: Target x position
    :param target_y: Target y position
    :param P: Image size
    :return: PxP numpy array of the Gaussian spot
    """
    height = torch.tensor([[[[height]]]])  # (1,1,1,1)
    width = torch.tensor([[[[width]]]])
    x = torch.tensor([[[[x]]]])
    y = torch.tensor([[[[y]]]])
    target_locs = torch.tensor([[[[target_x, target_y]]]])  # (1,1,1,1,2)
    spot = gaussian_spots(height, width, x, y, target_locs, P)
    return spot.squeeze().numpy()