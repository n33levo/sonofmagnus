from http.server import BaseHTTPRequestHandler, HTTPServer
import ssl
import subprocess
import threading
import urllib.error
import urllib.request

import pytest

from chessio.tls import ssl_context


def test_platform_context_requires_valid_certificates_and_hostnames():
    context = ssl_context()
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    assert context is ssl_context()


def test_nonmac_platform_does_not_require_a_keychain(monkeypatch):
    ssl_context.cache_clear()
    monkeypatch.setattr('chessio.tls.sys.platform', 'linux')
    def unavailable(*args, **kwargs):
        raise AssertionError('No macOS keychain on Linux')
    monkeypatch.setattr('chessio.tls.subprocess.check_output', unavailable)
    try:
        assert ssl_context().verify_mode == ssl.CERT_REQUIRED
    finally:
        ssl_context.cache_clear()


def test_untrusted_https_server_is_rejected_before_headers_are_sent(tmp_path):
    certificate, key = tmp_path / 'certificate.pem', tmp_path / 'key.pem'
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
        '-keyout', str(key), '-out', str(certificate), '-days', '1', '-subj', '/CN=localhost'],
        check=True, capture_output=True)
    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append(dict(self.headers))
            self.send_response(200)
            self.end_headers()
        def log_message(self, *args):
            pass
    server = HTTPServer(('127.0.0.1', 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    request = urllib.request.Request(f'https://localhost:{server.server_port}/',
                                     headers={'Authorization': 'Bearer fake-token'})
    try:
        with pytest.raises(urllib.error.URLError) as error:
            urllib.request.urlopen(request, timeout=3, context=ssl_context())
        assert isinstance(error.value.reason, ssl.SSLCertVerificationError)
        assert received == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
