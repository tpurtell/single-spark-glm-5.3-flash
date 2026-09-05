#!/usr/bin/env python3
"""Retain the full memory accounting behind the single-Spark cache budget."""
import sys
from pathlib import Path

path = Path(sys.argv[1]) / 'v1/worker/gpu_worker.py'
source = path.read_text()
old = '        logger.debug(profile_result)\n'
new = '        logger.info("GLM53 memory accounting: %s", profile_result)\n'
if new not in source:
    if source.count(old) != 1:
        raise RuntimeError('Memory accounting source drift')
    source = source.replace(old, new)
anchor = '''        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
'''
addition = '''        # Optional diagnostic only; never enabled in the release defaults.
        memory_trace = __import__("os").environ.get("GLM53_MEMORY_TRACE") == "1"
        if memory_trace:
            torch.cuda.memory._record_memory_history(max_entries=100000, stacks="python")

'''
if addition not in source:
    if source.count(anchor) != 1:
        raise RuntimeError('Memory trace start source drift')
    source = source.replace(anchor, addition + anchor)
anchor = '''        # Profile CUDA graph memory if graphs will be captured.
'''
addition = '''        if memory_trace:
            torch.cuda.memory._dump_snapshot("/root/.cache/glm53/profile-memory.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)
        logger.info("GLM53 memory snapshots: before=%s; after=%s",
                    profile_result.before_profile, profile_result.after_profile)

'''
if addition not in source:
    if source.count(anchor) != 1:
        raise RuntimeError('Memory trace end source drift')
    source = source.replace(anchor, addition + anchor)
compile(source, str(path), 'exec')
path.write_text(source)
print('Single-Spark memory accounting receipt enabled')
