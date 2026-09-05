#!/usr/bin/env python3
"""Admit GLM5.3's padded 2176-wide candidate table to the existing MG kernel.

All 2051 candidates are preserved; per-row lengths still bound the 125 padding
entries. This changes the dispatch capacity gate, not attention math.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1]) / 'b12x/attention/_shared/mla/prefill.py'
source = path.read_text()
replacements = [
    ('glm_topk_supported = topk in (512, 1024, 2048) or (',
     'glm_topk_supported = topk in (512, 1024, 2048, 2176) or ('),
    ('if _mg_nvfp4 and topk in (128, 512, 1024, 2048):',
     'if _mg_nvfp4 and topk in (128, 512, 1024, 2048, 2176):'),
    ('GLM_NSA topk in {512, 1024, 2048}; GLM_NEXT topk in ',
     'GLM_NSA topk in {512, 1024, 2048, 2176}; GLM_NEXT topk in '),
    ('NVFP4 (GLM-family, scale_format=2) topk in {128, 512, 1024, 2048}. ',
     'NVFP4 (GLM-family, scale_format=2) topk in {128, 512, 1024, 2048, 2176}. '),
]
for old, new in replacements:
    if source.count(old) != 1:
        raise RuntimeError(f'Expected one B12x source anchor: {old}')
    source = source.replace(old, new, 1)
path.write_text(source)
print('B12x GLM 2176-wide prefill capacity admitted')
