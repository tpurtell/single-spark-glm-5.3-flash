#!/usr/bin/env python3
"""Keep GLM's convolution verify width independent of compact state slots.

The upstream ReplaySSM series makes this correction for Qwen GDN and Mamba2,
but the GLM-specific KDA port also needs it. Mixed/eager batches expose a
one-column compact state table; pure-decode graph metadata broadcasts that
column into a preallocated K+1 table, hiding the error in short decode tests.
"""
import sys
from pathlib import Path

OLD = '            conv_mql = spec_state_indices_tensor.size(-1)\n'
NEW = (
    '            # Compact rollback stores one state slot, not one query token.\n'
    '            # The convolution must still process the entire verify window.\n'
    '            conv_mql = (\n'
    '                self.num_spec + 1\n'
    '                if self.use_replayssm_spec\n'
    '                else spec_state_indices_tensor.size(-1)\n'
    '            )\n'
)


def main(root):
    path = root / 'models/glm5next/nvidia/kda.py'
    text = path.read_text()
    if NEW in text:
        print('[skip] GLM ReplaySSM convolution verify width')
        return
    if text.count(OLD) != 1:
        raise RuntimeError('expected one GLM convolution max-query-length anchor')
    text = text.replace(OLD, NEW, 1)
    compile(text, str(path), 'exec')
    path.write_text(text)
    print('[ok] GLM ReplaySSM convolution uses K+1 tokens, not compact state columns')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
