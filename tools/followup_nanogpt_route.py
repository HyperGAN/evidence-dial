"""Load the owner's literal NanoGPT assignment without running shell startup."""
import os
from pathlib import Path
import re
import shlex
import sys


def load_key():
    values = []
    for line in (Path.home() / '.zshrc').read_text().splitlines():
        match = re.match(r'^\s*(?:export\s+)?NANOGPT_API_KEY\s*=\s*(.*?)\s*$', line)
        if match:
            words = shlex.split(match[1], comments=True, posix=True)
            if len(words) != 1 or any(c in words[0] for c in ['$','`','\n','\r']):
                raise RuntimeError('NanoGPT assignment must be a literal; shell startup was not executed')
            values.append(words[0])
    if not values:
        raise RuntimeError('No literal NanoGPT assignment in owner .zshrc')
    return values[-1]


if __name__ == '__main__':
    command = sys.argv[1:]
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        raise SystemExit('Supply a child command; credentials are never displayed')
    env = dict(os.environ, NANOGPT_API_KEY=load_key())
    os.execvpe(command[0], command, env)
