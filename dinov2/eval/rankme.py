import torch


def rankMe(z: torch.Tensor, epsilon=1e-7):
    """
    param z: a tensor of shape (B, N, K), where B is batch size.
    """
    assert not torch.isnan(z).any()
    assert not torch.isinf(z).any()
    assert z.ndim == 2, "Matrix is not 2-dimensional"

    z = z.float()  # convert to full precision
    singular_values = torch.linalg.svd(z, full_matrices=False).S
    sum_singular_values = torch.sum(singular_values, dim=-1, keepdim=True)

    pk = singular_values / (sum_singular_values + epsilon)
    log_pk = torch.log(pk + epsilon)

    score = torch.exp(-torch.sum(pk * log_pk, dim=-1))

    return score


if __name__ == "__main__":
    t = torch.randn((64, 10, 5))
    print(rankMe(t))
