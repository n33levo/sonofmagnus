from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import struct
import threading

import chess
import pytest

from model.artifacts import artifact_info
from scripts.fetch_expert import collect, decode
from scripts.check_expert import check


@pytest.mark.parametrize('failure', [None, 'changed', 'ignored'])
def test_bag_ranges_preserve_legal_labels_and_exclude_duplicates_and_tests(tmp_path, failure):
    root = chess.Board()
    second = root.copy()
    second.push_uci('e2e4')
    castle = chess.Board('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1')
    promotion = chess.Board('r6k/1P6/8/8/8/8/8/2K5 w - - 0 1')
    items = [(root, 'e2e4'), (root, 'd2d4'), (second, 'e7e5'), (castle, 'e1g1'), (promotion, 'b7b8q')]
    data, ends = b'', []
    for board, move in items:
        fen = board.fen().encode()
        assert len(fen) < 128
        data += bytes([len(fen)]) + fen + move.encode()
        ends.append(len(data))
    bag = data + struct.pack(f'<{len(ends)}Q', *ends)

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_response(200)
            self.send_header('Content-Length', str(len(bag)))
            self.send_header('ETag', 'frozen')
            self.end_headers()

        def do_GET(self):
            assert self.headers['If-Match'] == 'frozen'
            start, end = map(int, self.headers['Range'][6:].split('-'))
            chunk = bag[start:end + 1]
            self.send_response(200 if failure == 'ignored' else 206)
            self.send_header('ETag', 'changed' if failure == 'changed' else 'frozen')
            self.send_header('Content-Range', f'bytes {start}-{end}/{len(bag)}')
            self.send_header('Content-Length', str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)

        def log_message(self, *args):
            pass

    heldout = tmp_path / 'test.jsonl'
    heldout.write_text(json.dumps({'fen': second.fen()}) + '\n')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / 'expert.jsonl'
    try:
        url = f'http://127.0.0.1:{server.server_port}/dataset'
        if failure:
            with pytest.raises(ValueError, match='changed|refused'):
                collect(output, 3, [heldout], url, chunk_size=2)
            assert not output.exists()
        else:
            result = collect(output, 3, [heldout], url, chunk_size=2)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            assert [row['move'] for row in rows] == ['e2e4', 'e1g1', 'b7b8q']
            assert result['duplicates_excluded'] == result['heldout_excluded'] == 1
            assert result['source_records_scanned'] == 5 and result['artifact'] == artifact_info(output)
            assert all('value' not in row and 'q_values' not in row for row in rows)
            assert check(output)['legal_teacher_moves'] == 3
            with pytest.raises(FileExistsError):
                collect(output, 1, url=url)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_expert_decoder_rejects_an_illegal_teacher_move():
    fen = chess.STARTING_FEN.encode()
    with pytest.raises(ValueError, match='expert'):
        decode(bytes([len(fen)]) + fen + b'e2e5')
