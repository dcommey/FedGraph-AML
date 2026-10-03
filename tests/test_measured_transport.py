"""Bounded framing, integrity and context reject malformed probe traffic."""
import socket
import struct
import numpy as np
import pytest
from federated.measured_transport import encode,receive,context,validate_context,MAX_BYTES


def roundtrip(packet):
    a,b=socket.socketpair()
    try:
        a.sendall(packet);a.shutdown(socket.SHUT_WR)
        return receive(b)
    finally:a.close();b.close()


def test_numeric_frame_and_context():
    header=context('trial',2,'owner-2','coordinator','embedding')
    packet=encode(header,{'ids':np.array([1,3],dtype=np.int64),'h':np.ones((2,4),dtype=np.float32)})
    h,a,n=roundtrip(packet);validate_context(h,'trial',2,'owner-2','coordinator','embedding')
    assert n==len(packet) and np.array_equal(a['ids'],[1,3])
    for run,r,source,destination in [('other',2,'owner-2','coordinator'),('trial',3,'owner-2','coordinator'),('trial',2,'owner-0','coordinator'),('trial',2,'owner-2','owner-1')]:
        with pytest.raises(ValueError):validate_context(h,run,r,source,destination,'embedding')
    with pytest.raises(ValueError,match='Tampered'):roundtrip(packet[:-1]+bytes([packet[-1]^1]))


def test_frame_bounds_and_nonfinite():
    with pytest.raises(ValueError,match='frame length'):roundtrip(struct.pack('!I',MAX_BYTES+1))
    with pytest.raises(EOFError,match='Truncated'):roundtrip(struct.pack('!I',8)+b'x')
    with pytest.raises(ValueError,match='numeric'):encode({}, {'h':np.array([object()])})
    with pytest.raises(ValueError,match='Nonfinite'):roundtrip(encode({}, {'h':np.array([np.nan],dtype=np.float32)}))
