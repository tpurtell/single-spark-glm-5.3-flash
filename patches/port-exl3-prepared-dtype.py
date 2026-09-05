#!/usr/bin/env python3
"""Keep uniform Trellis's FP16 prepared storage separate from BF16 inputs."""
import sys
from pathlib import Path

path = Path(sys.argv[1]) / 'model_executor/layers/quantization/exl3.py'
source = path.read_text()
start = source.index('    def _prepare_rank_sliced_weights(')
end = source.index('    def get_fused_moe_quant_config(', start)
section = source[start:end]
old = 'params_dtype=layer.exl3_params_dtype,'
new = 'params_dtype=torch.float16,'
if section.count(old) != 2 and section.count(new) != 2:
    raise RuntimeError('Uniform Trellis preparation source drift')
section = section.replace(old, new)
source = source[:start] + section + source[end:]
compile(source, str(path), 'exec')
path.write_text(source)
print('Uniform Trellis FP16 prepared-weight contract restored; mixed BF16 unchanged')
