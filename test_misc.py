import torch
from util.misc import nested_tensor_from_tensor_list

def collate_fn(batch):
    batch = list(zip(*batch))
    breakpoint()
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)

def gen_img(n):
    return torch.rand((3, 10, 10))

def gen_targets(n):
    return {
        "boxes": torch.rand((n,4)),
        "labels": torch.randint(low=0, high=12, size=(n,)),
        "area": torch.rand(n) * 200,
        "size": (300,300),
        "orig_size": (300, 300)
    }

batch = list()
for i in range(2):
    n = int(torch.randint(1, 10, (1,)))
    batch += [(gen_img(n), gen_targets(n))]

out = collate_fn(batch)
breakpoint()
    