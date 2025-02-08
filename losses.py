import torch

# https://arxiv.org/pdf/2301.11093
# multi-scale training loss seems make diffusion models converge faster,
# We find that losses for higher resolution are noisier on average, and
# we therefore decrease the relative weight of the loss as we
# increase the resolution.


def multi_scale_loss(
    x,
    y,
    loss=torch.nn.functional.mse_loss,
    scales=(1, 0.5, 0.25),
    weights=(0.5, 0.75, 1.0),
):
    assert len(scales) == len(weights)

    h, w = x.shape[-2:]
    acc_loss = 0
    weights = [w / sum(weights) for w in weights]

    for i, scale in enumerate(scales):

        x_ = torch.nn.functional.interpolate(x, scale_factor=scale, mode="bilinear")
        y_ = torch.nn.functional.interpolate(y, scale_factor=scale, mode="bilinear")

        acc_loss = weights[i] * loss(x_, y_) + acc_loss

    return acc_loss


def test_multi_scale_loss():
    x = torch.randn(1, 3, 128, 128)
    y = torch.randn(1, 3, 128, 128)

    multi_scale_loss(x, y)
