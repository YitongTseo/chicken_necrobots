# run me with sudo!


import usb.core
import usb.util
import time
import csv
from datetime import datetime
import sys

import numpy as np

POLL_INTERVAL = 0.25
DURATION = 2 * 3600
MEASUREMENT = 'MEAN'
CHANNEL = 2  # which scope channel currently has the signal of interest
OUTFILE = f"scope_log_{datetime.now():%Y%m%d_%H%M%S}.csv"

# GDS-2000A series: each vertical division spans 25 ADC codes.
# Verified empirically against :MEAS:MEAN? on this unit (GDS-2302A, V1.30).
CODES_PER_DIV = 25

dev = usb.core.find(idVendor=0x2184, idProduct=0x0013)
if dev is None:
    raise RuntimeError("Scope not found.")

dev.set_configuration(2)
cfg = dev.get_active_configuration()
intf = cfg[(1, 0)]

ep_out = usb.util.find_descriptor(
    intf, custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT)
ep_in = usb.util.find_descriptor(
    intf, custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN)

USBTimeout = getattr(usb.core, 'USBTimeoutError', None)


def _is_timeout(e):
    if USBTimeout is not None and isinstance(e, USBTimeout):
        return True
    s = str(e).lower()
    return 'timed out' in s or 'timeout' in s


def scpi_query(cmd, timeout_ms=2000):
    """Send a SCPI command and read the full reply, handling multi-packet responses."""
    ep_out.write(cmd.encode() + b'\n')

    buffer = bytearray()
    deadline = time.time() + (timeout_ms / 1000)

    while time.time() < deadline:
        try:
            chunk = ep_in.read(64, timeout=200)
            buffer.extend(chunk)
            # GDS terminates messages with \n (or sometimes \r\n)
            if b'\n' in buffer:
                break
        except usb.core.USBError as e:
            if _is_timeout(e):
                continue
            raise
    # print(bytes(buffer))
    return bytes(buffer).decode(errors='replace').strip().rstrip('\x00').rstrip('<')


def _read_until_idle(idle_ms=400, total_timeout_s=8.0):
    """Read from ep_in until idle (no chunk for idle_ms) or total_timeout_s elapses.
    Used for binary blobs where we can't rely on a newline terminator."""
    buf = bytearray()
    deadline = time.time() + total_timeout_s
    while time.time() < deadline:
        try:
            chunk = ep_in.read(4096, timeout=idle_ms)
            if chunk:
                buf.extend(chunk)
            else:
                if buf:
                    break
        except usb.core.USBError as e:
            if _is_timeout(e):
                if buf:
                    break
                continue
            raise
    return bytes(buf)


def read_waveform_mean_v(channel):
    """Pull :ACQuire<n>:MEMory?, parse the ASCII header for Vertical Scale,
    decode int16 big-endian samples, and return the host-computed mean voltage.
    Bypasses the rounded resolution of :MEAS:MEAN?."""
    ep_out.write(f':ACQuire{channel}:MEMory?\n'.encode())
    raw = _read_until_idle(idle_ms=400, total_timeout_s=8.0)
    hash_idx = raw.find(b'#')
    if hash_idx < 0:
        raise RuntimeError(f'no data block marker in CH{channel} memory reply ({len(raw)} bytes)')

    header = raw[:hash_idx].decode(errors='replace')
    vscale = None
    for entry in header.split(';'):
        if ',' not in entry:
            continue
        k, v = entry.split(',', 1)
        if k.strip() == 'Vertical Scale':
            vscale = float(v)
            break
    if vscale is None:
        raise RuntimeError(f'Vertical Scale not found in header: {header!r}')

    # IEEE-488.2 block: '#' + N (1 digit) + N digits = byte_count + <binary>
    n_digits = int(chr(raw[hash_idx + 1]))
    byte_count = int(raw[hash_idx + 2:hash_idx + 2 + n_digits].decode())
    data_start = hash_idx + 2 + n_digits
    data = raw[data_start:data_start + byte_count]
    if len(data) < byte_count:
        raise RuntimeError(f'truncated data block: got {len(data)} of {byte_count} bytes')

    samples = np.frombuffer(data, dtype='>i2')
    return float(samples.mean()) * (vscale / CODES_PER_DIV)

print("Connected to:", scpi_query('*IDN?'))
print(f"Logging to: {OUTFILE}")
print(f"Press Ctrl+C to stop early.\n")

start = time.time()
sample_count = 0

with open(OUTFILE, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['elapsed_s', 'timestamp', f'ch1_{MEASUREMENT}_V', f'ch2_{MEASUREMENT}_V'])
    f.flush()
    
    try:
        while (time.time() - start) < DURATION:
            loop_start = time.time()
            try:
                ch1 = 0.0
                ch2 = 0.0
                v = read_waveform_mean_v(CHANNEL)
                if CHANNEL == 1:
                    ch1 = v
                else:
                    ch2 = v
                elapsed = time.time() - start
                w.writerow([f'{elapsed:.4f}', datetime.now().isoformat(),
                            f'{ch1:.6f}', f'{ch2:.6f}'])
                f.flush()
                sample_count += 1

                # Print a status line every 50 samples
                if sample_count % 50 == 0:
                    print(f'{elapsed:7.1f}s  CH1={ch1:+.6f}V  CH2={ch2:+.6f}V  ({sample_count} samples)')
            except Exception as e:
                print(f'[{time.time()-start:.1f}s] error: {e}', file=sys.stderr)
            
            sleep_time = POLL_INTERVAL - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print(f'\nStopped at {time.time()-start:.1f}s with {sample_count} samples.')

print(f'\nDone. {sample_count} samples written to {OUTFILE}')