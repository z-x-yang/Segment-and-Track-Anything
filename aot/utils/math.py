import torch


def generate_permute_matrix(dim, num, keep_first=True, gpu_id=0):
    all_matrix = []
    for idx in range(num):
        random_matrix = torch.eye(dim, device=torch.device('cuda', gpu_id))
        if keep_first:
            fg = random_matrix[1:][torch.randperm(dim - 1)]
            random_matrix = torch.cat([random_matrix[0:1], fg], dim=0)
        else:
            random_matrix = random_matrix[torch.randperm(dim)]
        all_matrix.append(random_matrix)
    return torch.stack(all_matrix, dim=0)


def truncated_normal_(tensor, mean=0, std=.02):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4, )).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor
