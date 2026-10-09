"""Deny real broker/network paths; tests must inject explicit inert transports."""
ATTEMPTS = []
_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import asyncio
    import socket
    import smtplib
    import urllib.request
    import sys
    import requests
    from ib_insync import IB
    from ib_insync.client import Client
    from ib_insync.connection import Connection

    def denied(*args, **kwargs):
        ATTEMPTS.append('native broker/network path attempted')
        raise AssertionError('Offline validation forbids broker/network operations')

    async def denied_async(*args, **kwargs):
        return denied(*args, **kwargs)

    IB.connect = denied
    IB.connectAsync = denied_async
    IB.placeOrder = denied
    IB.cancelOrder = denied
    Client.connect = denied
    Client.connectAsync = denied_async
    Client.send = denied
    Connection.connectAsync = denied_async
    asyncio.open_connection = denied_async
    asyncio.start_server = denied_async
    socket.create_connection = denied
    native_connect = socket.socket.connect
    socketpair_code = getattr(socket.socketpair, '__code__', None)
    def guarded_connect(sock, address):
        # Windows asyncio uses the stdlib socketpair fallback for its private
        # wakeup pipe. Permit only that exact stdlib call site; broker and other
        # arbitrary loopback/remote connects stay denied.
        if socketpair_code is not None and sys._getframe(1).f_code is socketpair_code:
            return native_connect(sock, address)
        return denied(sock, address)
    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = denied
    requests.sessions.Session.request = denied
    smtplib.SMTP.__init__ = denied
    urllib.request.urlopen = denied
    _INSTALLED = True
