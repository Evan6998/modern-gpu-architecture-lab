from assignments.a2_ampere.reference import gemm


def cluster_copy(x):
    return x.unsqueeze(0).expand(2, -1, -1).clone()
