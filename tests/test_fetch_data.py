import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest

from scripts.fetch_data import fetch


@pytest.mark.parametrize('body', [b'frozen data', b'broken data', b'frozen data EXTRA', b'short'])
def test_verified_download_never_publishes_corrupt_or_overwrites_existing(tmp_path, body):
    expected = b'frozen data'
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    entry = {'url': f'http://127.0.0.1:{server.server_port}/data', 'bytes': len(expected),
             'sha256': hashlib.sha256(expected).hexdigest()}
    try:
        output = tmp_path / 'test.jsonl'
        if body == expected:
            assert fetch(output.name, entry, tmp_path).read_bytes() == expected
            assert fetch(output.name, entry, tmp_path) == output
            assert len(requests) == 1
        else:
            with pytest.raises(ValueError, match='checksum mismatch'):
                fetch(output.name, entry, tmp_path)
            assert not output.exists()
        assert not list(tmp_path.glob('*.part'))
        output.write_bytes(b'active training file')
        with pytest.raises(ValueError, match='differs'):
            fetch(output.name, entry, tmp_path)
        assert output.read_bytes() == b'active training file'
        assert len(requests) == 1
        with pytest.raises(ValueError, match='file name'):
            fetch('../escape', entry, tmp_path)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_optional_archive_requires_an_explicit_name(monkeypatch):
    from scripts.fetch_data import main
    seen = []
    monkeypatch.setattr('scripts.fetch_data.fetch', lambda name, entry, directory: seen.append(name) or name)
    monkeypatch.setattr('sys.argv', ['fetch_data'])
    main()
    assert len(seen) == 7 and 'lichess_db_puzzle.csv.zst' not in seen
    seen.clear()
    monkeypatch.setattr('sys.argv', ['fetch_data', 'lichess_db_puzzle.csv.zst'])
    main()
    assert seen == ['lichess_db_puzzle.csv.zst']
