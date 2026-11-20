import argparse
import getpass
import json
import os
from pathlib import Path
import urllib.request

from chessio.tls import ssl_context


TOKEN_FILE = Path.home() / '.config/sonofmagnus/lichess.token'


def save_token(path=TOKEN_FILE):
    path = Path(path)
    token = getpass.getpass('Paste Lichess token (hidden): ').strip()
    if not token or any(char.isspace() for char in token):
        raise ValueError('Invalid token')
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as stream:
        stream.write(token + '\n')
    print(f'Token saved to {path}; its contents will not be printed')


def account(token, expected):
    if not token or any(char.isspace() for char in token):
        raise ValueError('Invalid token')
    request = urllib.request.Request('https://lichess.org/api/account',
                                     headers={'Authorization': 'Bearer ' + token})
    with urllib.request.urlopen(request, timeout=30, context=ssl_context()) as response:
        profile = json.load(response)
    if profile['id'].lower() != expected.lower():
        raise ValueError('Token belongs to a different Lichess account')
    return profile


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save or verify local Lichess BOT access')
    parser.add_argument('--save-token', action='store_true')
    parser.add_argument('--token-file', type=Path, default=TOKEN_FILE)
    parser.add_argument('--account')
    parser.add_argument('--upgrade', action='store_true', help='Permanently convert the specified unused account to BOT')
    args = parser.parse_args()
    if args.save_token:
        if args.upgrade or args.account:
            parser.error('Save the token separately before checking or upgrading')
        save_token(args.token_file)
    else:
        if not args.account:
            parser.error('--account is required')
        token = args.token_file.read_text().strip()
        profile = account(token, args.account)
        if args.upgrade and profile.get('title') != 'BOT':
            if profile.get('count', {}).get('all') != 0:
                parser.error('Only an unused account can be converted to BOT')
            request = urllib.request.Request('https://lichess.org/api/bot/account/upgrade', data=b'',
                                             headers={'Authorization': 'Bearer ' + token})
            with urllib.request.urlopen(request, timeout=30, context=ssl_context()) as response:
                if json.load(response).get('ok') is not True:
                    raise ValueError('BOT upgrade failed')
            profile = account(token, args.account)
        print(json.dumps({key: profile.get(key) for key in ('id', 'username', 'title', 'perfs')}, indent=2))
