import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import urllib.request

import chess

from chessio.records import holdout_positions
from model.artifacts import artifact_info

URL = 'https://storage.googleapis.com/searchless_chess/data/train/behavioral_cloning_data.bag'


def byte_range(url, start, end, identity, size):
    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}', 'If-Match': identity})
    with urllib.request.urlopen(request, timeout=60) as response:
        if (response.status != 206 or response.headers.get('ETag') != identity
                or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}'):
            raise ValueError('Remote dataset changed or exact byte range refused')
        data = response.read(end - start + 2)
    if len(data) != end - start + 1:
        raise ValueError('Incomplete dataset range')
    return data


def decode(record):
    length = shift = cursor = 0
    while cursor < len(record) and shift < 35:
        byte = record[cursor]
        length |= (byte & 127) << shift
        cursor += 1
        if byte < 128:
            break
        shift += 7
    else:
        raise ValueError('Invalid FEN length')
    board = chess.Board(record[cursor:cursor + length].decode('utf-8'))
    move = chess.Move.from_uci(record[cursor + length:].decode('utf-8'))
    if not board.is_valid() or move not in board.legal_moves:
        raise ValueError('Invalid expert position or move')
    return board, move


def collect(output, positions, excluded=(), url=URL, chunk_size=100000):
    output = Path(output)
    part, database = output.with_suffix('.jsonl.part'), output.with_suffix('.dedup.sqlite')
    if positions < 1 or chunk_size < 1:
        raise ValueError('Positive corpus and chunk sizes required')
    if any(path.exists() for path in (output, part, database, output.with_suffix('.json'))):
        raise FileExistsError('Use a new output path')
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = {str(path): artifact_info(path) for path in excluded}
    blocked = holdout_positions(excluded)
    producer = artifact_info(__file__)
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as response:
        size, identity = int(response.headers['Content-Length']), response.headers['ETag']
    index = struct.unpack('<Q', byte_range(url, size - 8, size - 1, identity, size))[0]
    if not 0 < index < size or (size - index) % 8:
        raise ValueError('Invalid BAG index')
    records, scanned, accepted, duplicates, heldout, offset = (size - index) // 8, 0, 0, 0, 0, 0
    phases = dict.fromkeys((0, 1, 2, 3), 0)
    downloaded, digest = 8, hashlib.sha256()
    with sqlite3.connect(database) as unique, part.open('x') as saved:
        unique.execute('CREATE TABLE positions (fen TEXT PRIMARY KEY) WITHOUT ROWID')
        for start in range(0, records, chunk_size):
            count = min(chunk_size, records - start)
            limits = byte_range(url, index + start * 8, index + (start + count) * 8 - 1, identity, size)
            ends = struct.unpack(f'<{count}Q', limits)
            if not offset < ends[0] or ends[-1] > index or any(a >= b for a, b in zip(ends, ends[1:])):
                raise ValueError('Invalid record boundaries')
            data = byte_range(url, offset, ends[-1] - 1, identity, size)
            downloaded += len(data) + len(limits)
            digest.update(limits + data)
            cursor = 0
            for end in ends:
                length = end - offset
                board, move = decode(data[cursor:cursor + length])
                cursor += length
                offset = end
                scanned += 1
                key = ' '.join(board.fen().split()[:4])
                if key in blocked:
                    heldout += 1
                    continue
                if not unique.execute('INSERT OR IGNORE INTO positions VALUES (?)', (key,)).rowcount:
                    duplicates += 1
                    continue
                saved.write(json.dumps({'fen': board.fen(), 'move': move.uci(), 'weight': 3.,
                                        'source': 'chessbench-stockfish-top1'}, separators=(',', ':')) + '\n')
                pieces = len(board.piece_map())
                phases[0 if pieces >= 24 else 1 if pieces >= 16 else 2 if pieces >= 8 else 3] += 1
                accepted += 1
                if accepted == positions:
                    break
            unique.commit()
            print(json.dumps({'positions': accepted, 'source_records': scanned}), flush=True)
            if accepted == positions:
                break
    if accepted != positions or producer != artifact_info(__file__) or any(artifact_info(path) != info for path, info in sources.items()):
        raise ValueError('Incomplete corpus or changed source')
    part.replace(output)
    database.unlink()
    result = dict(positions=accepted, source=url, source_bytes=size, source_etag=identity, source_records_scanned=scanned,
                  downloaded_bytes=downloaded, fetched_ranges_sha256=digest.hexdigest(), duplicates_excluded=duplicates,
                  heldout_excluded=heldout, pieces_phases=phases, excluded_data=sources, producer=producer,
                  artifact=artifact_info(output), license='CC-BY-4.0; retain ChessBench attribution',
                  selection='First distinct eligible positions in the upstream training file; not a uniform reservoir',
                  labels='Stockfish best moves only; no action-value, position-value or per-row depth labels provided')
    output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--positions', type=int, default=2500000)
    parser.add_argument('--exclude-data', required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.out, args.positions, args.exclude_data.split(',')), indent=2))
