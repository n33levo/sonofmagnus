import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', default='data/arena_pool.json')
    parser.add_argument('--cc', default='clang')
    args = parser.parse_args()
    for engine in json.loads(Path(args.pool).read_text())['engines']:
        output = Path(engine['path']).resolve()
        source = output.parent / (engine['name'] + '-source')
        output.parent.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            subprocess.run(['git', 'clone', '--depth', '1', '--branch', engine['tag'], engine['source'], str(source)], check=True)
        revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != engine['revision']:
            raise ValueError(f'Wrong source revision for {engine["name"]}: {revision}')
        subprocess.run(['git', '-C', str(source), 'diff', '--quiet', 'HEAD'], check=True)
        if engine['build'] == 'stash':
            command, binary = ['make', '-j4', f'CC={args.cc}', 'ARCH=generic'], 'stash-bot'
        elif engine['build'] == 'weiss':
            command, binary = ['make', 'bench-basic', f'COMP={args.cc}'], 'weiss'
        else:
            raise ValueError('Unknown opponent build')
        with output.with_suffix('.build.log').open('w') as log:
            subprocess.run(command, cwd=source / 'src', stdout=log, stderr=subprocess.STDOUT, check=True)
        shutil.copy2(source / 'src' / binary, output)
        print(f'{engine["name"]}: {revision}', flush=True)


if __name__ == '__main__':
    main()
