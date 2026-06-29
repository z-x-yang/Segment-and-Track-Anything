import torch

def generate_permute_matrix(dim, num, keep_first=True, device=None, gpu_id=None):
    # Backward-compatible path: existing callers may still pass gpu_id=...
    if device is None:
        if gpu_id is not None and torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            
    all_matrix = []
    for _ in range(num):
        random_matrix = torch.eye(dim, device=device)

        if keep_first:
            perm = torch.randperm(dim - 1, device=device)
            fg = random_matrix[1:][perm]
            random_matrix = torch.cat([random_matrix[:1], fg], dim=0)
        else:
            perm = torch.randperm(dim, device=device)
            random_matrix = random_matrix[perm]

        all_matrix.append(random_matrix)

    return torch.stack(all_matrix, dim=0)


def truncated_normal_(tensor, mean=0., std=0.02):
    with torch.no_grad():
        size = tensor.shape
        tmp = torch.randn(size + (4,), device=tensor.device)
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True).indices
        tensor.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.mul_(std).add_(mean)
    return tensor

