#!/usr/bin/env python3
"""Use reachable GLM sparse-attention bounds, without bypassing memory accounting."""
import sys
from pathlib import Path


def replace(path, old, new):
    source = path.read_text()
    if new in source:
        return
    if source.count(old) != 1:
        raise RuntimeError(f'{path}: sparse-memory source drift')
    source = source.replace(old, new)
    compile(source, str(path), 'exec')
    path.write_text(source)


root = Path(sys.argv[1])
replace(root / 'v1/attention/backends/mla/indexer.py',
'''    return max_model_len * 40
''',
'''    hf_config = vllm_config.model_config.hf_text_config
    if getattr(hf_config, "model_type", None) in {"glm5_next", "glm5_next_text"}:
        # The same helper sizes BOTH the model's actual gather allocation and
        # the metadata chunk planner. A scheduled batch cannot contain more
        # than max_num_seqs full-length contexts. Keep the token-granular
        # upper bound (conservative for kpool) and the original 40-context cap.
        return max_model_len * min(40, vllm_config.scheduler_config.max_num_seqs)
    return max_model_len * 40
''')
path = root / 'model_executor/layers/attention/mla_attention.py'
old = '''            _ = torch.empty(
                (
                    self.chunked_prefill_workspace_size,
                    self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                ),
                device=k_c_normed.device,
                dtype=k_c_normed.dtype,
            )
'''
new = '''            if self.attn_backend.get_name() == "B12X_MLA_SPARSE":
                # B12x never calls dense _compute_prefill_context or expands
                # the entire context through kv_b_proj. Its real caller-owned
                # query/decode/extend scratch was reserved at construction;
                # borrow it here too so the profiler sees that exact capacity.
                self.impl._borrow_workspaces()
            else:
''' + ''.join('    ' + line if line.strip() else line for line in old.splitlines(True))
replace(path, old, new)
replace(root / 'v1/worker/gpu/model_runner.py',
'''        min_blocks = self.compilation_config.max_cudagraph_capture_size or 1
''',
'''        # Even a small capture ladder needs one KDA state per scheduler slot.
        min_blocks = max(self.compilation_config.max_cudagraph_capture_size or 1,
                         self.scheduler_config.max_num_seqs)
''')
print('GLM gather capacity and B12x sparse-only memory profiling applied')
