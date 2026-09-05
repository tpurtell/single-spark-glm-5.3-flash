#!/usr/bin/env python3
"""Size metadata from the model's physical selection buffer, not index_topk."""
import sys
from pathlib import Path

path = Path(sys.argv[1]) / 'v1/attention/backends/mla/b12x_mla_sparse.py'
source = path.read_text()
old = '        self.topk_tokens = vllm_config.model_config.hf_config.index_topk\n'
new = old + '''        # GLM kpool adds its trailing tokens and pads the shared index
        # buffer to 128 columns. The model/impl owns that width (2176, not
        # config.index_topk=2048). All graph-stable conversion buffers must
        # carry the complete physical selection table, including the tail.
        context = vllm_config.compilation_config.static_forward_context
        widths = {
            int(layer.impl.topk_tokens)
            for name in layer_names
            if (layer := context.get(name)) is not None
            and hasattr(getattr(layer, "impl", None), "topk_tokens")
        }
        if len(widths) > 1:
            raise ValueError(f"MLA cache group has inconsistent index widths: {widths}")
        if widths:
            self.topk_tokens = widths.pop()
'''
if new not in source:
    if source.count(old) != 1:
        raise RuntimeError('Sparse MLA metadata builder source drift')
    source = source.replace(old, new)
    compile(source, str(path), 'exec')
    path.write_text(source)
print('GLM graph metadata uses the complete model-owned top-k buffer width')
