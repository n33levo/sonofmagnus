from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import threading

import pytest
import zstandard

from chessio.remote import RangeReader
from scripts import build_human_corpus


@pytest.mark.parametrize('failure', ['disconnect', 'changed', 'ignored'])
def test_compressed_archive_recovery_verifies_the_exact_range(monkeypatch, failure):
    plain = b'positions and complete move histories\n' * 1000
    packed = zstandard.ZstdCompressor().compress(plain)
    requested = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requested.append(self.headers.get('Range'))
            offset = int(self.headers['Range'][6:-1]) if self.headers.get('Range') else 0
            self.send_response(200 if not offset or failure == 'ignored' else 206)
            self.send_header('ETag', 'changed' if offset and failure == 'changed' else 'frozen')
            self.send_header('Content-Length', str(len(packed) - offset))
            if offset:
                self.send_header('Content-Range', f'bytes {offset}-{len(packed) - 1}/{len(packed)}')
            self.end_headers()
            self.wfile.write(packed[offset:] if offset else packed[:17])
            self.wfile.flush()
            self.close_connection = True

        def log_message(self, *args):
            pass

    monkeypatch.setattr('chessio.remote.time.sleep', lambda _: None)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with RangeReader(f'http://127.0.0.1:{server.server_port}/archive') as source:
            with zstandard.ZstdDecompressor().stream_reader(source) as decoded:
                if failure == 'disconnect':
                    assert decoded.read() == plain
                    assert source.position == len(packed)
                else:
                    with pytest.raises(ValueError, match='changed|refused'):
                        decoded.read()
        assert requested == [None, 'bytes=17-']
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize('corrupt', [False, True])
def test_corpus_recovery_verifies_previous_rows_before_publishing(tmp_path, monkeypatch, corrupt):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / 'data'
    data.mkdir()
    for name in ('puzzles_holdout', 'puzzles_final3000', 'stockfish_holdout', 'eval_holdout', 'endgame_holdout'):
        (data / (name + '.jsonl')).write_text('')
    pgn = '[Event "Rated Blitz game"]\n[Site "https://lichess.org/test"]\n[WhiteElo "2400"]\n[BlackElo "2400"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 1-0\n'
    packed = zstandard.ZstdCompressor().compress(pgn.encode())

    class Source(io.BytesIO):
        headers = {'ETag': 'frozen'}

    monkeypatch.setattr(build_human_corpus, 'RangeReader', lambda _: Source(packed))
    out = data / 'train.jsonl'
    part = data / 'train.jsonl.part'
    prefix = json.dumps({'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
        'move': 'd2d4' if corrupt else 'e2e4', 'value': 1., 'game_id': 'https://lichess.org/test', 'weight': 2.}, separators=(',', ':')) + '\n'
    part.write_text(prefix)
    monkeypatch.setattr('sys.argv', ['build', '--month', '2024-01', '--positions', '4', '--out', str(out), '--resume'])
    if corrupt:
        with pytest.raises(ValueError, match='does not match'):
            build_human_corpus.main()
        assert not out.exists()
        assert part.read_text() == prefix
    else:
        build_human_corpus.main()
        assert out.read_text().startswith(prefix)
        assert len(out.read_text().splitlines()) == 4
        assert not part.exists()
        assert json.loads(out.with_suffix('.json').read_text())['positions'] == 4
