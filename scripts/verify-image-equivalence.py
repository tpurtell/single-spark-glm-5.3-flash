#!/usr/bin/env python3
"""Prove a metadata-only release promotion retained the qualified runtime."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess


def inspect(reference):
    return json.loads(subprocess.check_output(
        ['docker', 'image', 'inspect', reference], text=True))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualified', required=True, help='Qualified immutable image ID')
    parser.add_argument('--release', required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = inspect(args.qualified)
    release = inspect(args.release)
    base_config = dict(base['Config'])
    release_config = dict(release['Config'])
    base_labels = base_config.pop('Labels', None) or {}
    release_labels = release_config.pop('Labels', None) or {}
    changed_labels = {
        key: {'qualified': base_labels.get(key), 'release': release_labels.get(key)}
        for key in sorted(set(base_labels) | set(release_labels))
        if base_labels.get(key) != release_labels.get(key)
    }
    checks = {
        'qualified_id_matches': base['Id'] == args.qualified,
        'same_rootfs': base['RootFS'] == release['RootFS'],
        'same_runtime_config': base_config == release_config,
        'same_platform': all(base.get(k) == release.get(k)
                             for k in ('Architecture', 'Os', 'Variant')),
        'arm64_linux': release['Architecture'] == 'arm64' and release['Os'] == 'linux',
        'source_revision_matches': release_labels.get('org.opencontainers.image.revision')
                                   == args.source_revision,
        'only_source_label_changed': set(changed_labels)
                                     <= {'org.opencontainers.image.revision'},
    }
    receipt = {
        'schema': 'glm53-metadata-only-image-promotion.v1',
        'captured_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'qualified_image_id': base['Id'], 'release_image_id': release['Id'],
        'release_reference': args.release, 'source_revision': args.source_revision,
        'checks': checks, 'changed_labels': changed_labels,
        'rootfs': release['RootFS'], 'passed': all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    if not receipt['passed']:
        raise SystemExit('Release image differs from the qualified runtime')


if __name__ == '__main__':
    main()
