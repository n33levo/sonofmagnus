import io
import json
import sys
import time
import urllib.request


class RangeReader(io.RawIOBase):
    def __init__(self, url, timeout=60, retries=5):
        super().__init__()
        self.url, self.timeout, self.retries = url, timeout, retries
        self.position = 0
        self.response = None
        self.headers = None
        self.total = None
        self.identity = None

    def readable(self):
        return True

    def connect(self):
        request = urllib.request.Request(self.url, headers={'Range': f'bytes={self.position}-'} if self.position else {})
        response = urllib.request.urlopen(request, timeout=self.timeout)
        headers = dict(response.headers)
        identity = (response.headers.get('ETag'), response.headers.get('Last-Modified'))
        if self.position:
            expected = f'bytes {self.position}-{self.total - 1}/{self.total}'
            if response.status != 206 or response.headers.get('Content-Range') != expected or identity != self.identity:
                response.close()
                raise ValueError('Remote archive changed or the server refused the exact resume range')
        else:
            if response.status != 200:
                response.close()
                raise ValueError('Expected a complete remote archive response')
            self.total = int(response.headers['Content-Length'])
            if not any(identity):
                response.close()
                raise ValueError('Remote archive needs an ETag or modification date for safe recovery')
            self.headers, self.identity = headers, identity
        self.response = response

    def read(self, size=-1):
        if size < 0:
            raise ValueError('Remote archives must be read in bounded chunks')
        if not size or self.position == self.total:
            return b''
        for attempt in range(self.retries + 1):
            try:
                if self.response is None:
                    self.connect()
                chunk = self.response.read1(size)
                if not chunk and self.position < self.total:
                    raise OSError('Remote archive ended before its declared byte count')
                self.position += len(chunk)
                return chunk
            except (OSError, EOFError) as error:
                if self.response is not None:
                    self.response.close()
                    self.response = None
                if attempt == self.retries:
                    raise
                print(json.dumps({'archive_retry': attempt + 1, 'byte': self.position, 'error': str(error)}), file=sys.stderr, flush=True)
                time.sleep(min(2 ** attempt, 8))

    def close(self):
        if self.response is not None:
            self.response.close()
        super().close()
