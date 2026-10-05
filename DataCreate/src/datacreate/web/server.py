"""Reserve the local GUI port before loading models or starting the server."""
from __future__ import annotations

import errno
import socket


def bind_gui_socket(port: int) -> socket.socket:
    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535.")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Windows must not allow a second listener to share this address.
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        listener.bind(("127.0.0.1", port))
        listener.listen(128)
        return listener
    except OSError as error:
        listener.close()
        if error.errno == errno.EADDRINUSE or getattr(error, "winerror", None) == 10048:
            raise SystemExit(
                f"Port {port} is already in use. No new GUI was started.\n"
                f"If this is your existing GUI, open http://127.0.0.1:{port}/\n"
                f"Practice studio: http://127.0.0.1:{port}/studio\n"
                "To start a separate GUI, choose an unused port:\n"
                "  datacreate serve --port <unused-port>\n"
                "To reload updated code, stop the existing server with Ctrl+C in its terminal, then start it again."
            ) from None
        raise
