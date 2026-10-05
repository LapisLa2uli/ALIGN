import socket

import pytest

from datacreate.web.server import bind_gui_socket


def test_busy_port_explains_how_to_open_existing_gui():
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        with pytest.raises(SystemExit) as error:
            bind_gui_socket(port)
        assert f"http://127.0.0.1:{port}/studio" in str(error.value)
        assert "--port" in str(error.value)
        # The existing server still owns its socket.
        assert occupied.getsockname()[1] == port


def test_free_socket_is_reserved_for_server():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with bind_gui_socket(port) as listener:
        assert listener.getsockname() == ("127.0.0.1", port)
        with pytest.raises(SystemExit):
            bind_gui_socket(port)


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_invalid_ports(port):
    with pytest.raises(ValueError, match="between 1 and 65535"):
        bind_gui_socket(port)
