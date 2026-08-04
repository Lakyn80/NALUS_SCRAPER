from __future__ import annotations

import ipaddress


def assert_local_bind(host: str) -> None:
    ip = ipaddress.ip_address(host)
    if not ip.is_loopback:
        raise ValueError(f"Parser review web UI may only bind to a loopback address, got {host}")
