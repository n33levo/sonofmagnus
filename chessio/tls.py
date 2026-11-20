from functools import cache
import ssl
import subprocess
import sys


@cache
def ssl_context():
    context = ssl.create_default_context()
    if sys.platform == 'darwin':
        roots = subprocess.check_output(['/usr/bin/security', 'find-certificate', '-a', '-p',
            '/System/Library/Keychains/SystemRootCertificates.keychain'], text=True, timeout=10)
        context.load_verify_locations(cadata=roots)
    return context
