#!/usr/bin/env python3
"""Bound InstantTensor's transient allocator footprint on unified-memory GB10."""
import sys
from pathlib import Path

path = Path(sys.argv[1]) / 'model_executor/model_loader/weight_utils.py'
source = path.read_text()
old = '''            for name, tensor in f.tensors():
                pbar.update(tensor.numel() * tensor.element_size())
                yield name, tensor
'''
new = '''            for tensor_index, (name, tensor) in enumerate(f.tensors()):
                pbar.update(tensor.numel() * tensor.element_size())
                yield name, tensor
                # GB10's CUDA and host allocations consume the same pool.
                # Keep copy=True (EXL3 may retain tensors), but reclaim dead
                # clones under pressure, as in the qualified MIA recipe.
                if (tensor_index & 255) == 255:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                    cached_bytes = (torch.cuda.memory_reserved(device)
                                    - torch.cuda.memory_allocated(device))
                    if free_bytes * 5 < total_bytes * 2 and cached_bytes > 1 << 30:
                        torch.cuda.empty_cache()
'''
if new not in source:
    if source.count(old) != 1:
        raise RuntimeError('InstantTensor streaming iterator source drift')
    source = source.replace(old, new)
    compile(source, str(path), 'exec')
    path.write_text(source)
print('InstantTensor pressure-triggered allocator trim enabled')
