"""GLM's visible-answer benchmark path, shared by the evaluation clients."""
import json
import urllib.request


def post(base_url, path, payload, timeout=3600):
    request = urllib.request.Request(
        base_url.rstrip('/') + path,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def visible_prompt(base_url, model, messages, timeout=3600):
    base_url = base_url.rstrip('/').removesuffix('/v1')
    rendered = post(base_url, '/v1/chat/completions/render', {
        'model': model, 'messages': messages, 'reasoning_effort': 'low',
    }, timeout)
    close = post(base_url, '/tokenize', {
        'model': model, 'prompt': '</think>', 'add_special_tokens': False,
    }, timeout)
    # The pinned GLM template opens <think> unconditionally and ignores the
    # DeepSeek/Qwen thinking flags. Use the same closed-think token sequence
    # as Brandon's qualified visible-answer benchmarks.
    return rendered['token_ids'] + close['tokens']
