# run me with sudo!
#
# Diagnostic: compare host-computed mean (from raw :ACQ:MEM? memory dump)
# against the scope's own :MEAS:MEAN? value, side by side, for N captures.
#
# Interpretation:
#   - host ~= meas (within a couple mV)        -> formula is correct.
#                                                  Any disagreement with an external
#                                                  meter is real (grounding, probe ratio,
#                                                  source drift, calibration).
#   - host - meas ~= constant nonzero offset   -> we're missing :CHAN:OFFSet? in the
#                                                  conversion. Fix in scope_logger.py.
#   - host wanders, meas stable                -> parser bug (alignment / endianness).

import usb.core
import usb.util
import time
import numpy as np

CHANNEL = 2
N_CAPTURES = 10
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


def read_all(idle_ms=400, total_timeout_s=8.0):
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


def query_text(cmd, idle_ms=200, total_timeout_s=2.0):
    ep_out.write(cmd.encode() + b'\n')
    return read_all(idle_ms=idle_ms, total_timeout_s=total_timeout_s).decode(errors='replace').strip().rstrip('\x00').rstrip('<')


def read_memory_block(channel):
    ep_out.write(f':ACQuire{channel}:MEMory?\n'.encode())
    raw = read_all(idle_ms=400, total_timeout_s=8.0)
    hash_idx = raw.find(b'#')
    if hash_idx < 0:
        raise RuntimeError(f'no data block marker ({len(raw)} bytes)')
    header = raw[:hash_idx].decode(errors='replace')
    n_digits = int(chr(raw[hash_idx + 1]))
    byte_count = int(raw[hash_idx + 2:hash_idx + 2 + n_digits].decode())
    data_start = hash_idx + 2 + n_digits
    data = raw[data_start:data_start + byte_count]
    samples = np.frombuffer(data, dtype='>i2')
    return header, samples


def parse_header_field(header, key):
    for entry in header.split(';'):
        if ',' not in entry:
            continue
        k, v = entry.split(',', 1)
        if k.strip() == key:
            return v.strip()
    return None


print("=" * 78)
print("IDN:", query_text('*IDN?'))
print("=" * 78)
print(f"\nChannel under test: CH{CHANNEL}")
print(f"\n--- Channel settings ---")
for q in [
    f':CHANnel{CHANNEL}:SCALe?',
    f':CHANnel{CHANNEL}:OFFSet?',
    f':CHANnel{CHANNEL}:POSition?',
    f':CHANnel{CHANNEL}:PROBe:RATio?',
    f':CHANnel{CHANNEL}:COUPling?',
    f':TIMebase:SCALe?',
]:
    print(f'  {q:40s} -> {query_text(q)!r}')

# Set MEAS source once.
query_text(f':MEASure:SOURce1 CH{CHANNEL}')

print(f"\n--- {N_CAPTURES} back-to-back captures ---")
print(f"{'#':>3}  {'host_mean (V)':>14}  {':MEAS:MEAN? (V)':>17}  "
      f"{'diff (mV)':>10}  {'n_samples':>9}  {'codes_min':>9}  {'codes_max':>9}  {'vscale':>8}  {'h_off':>10}")

host_means = []
meas_means = []
for i in range(N_CAPTURES):
    try:
        header, samples = read_memory_block(CHANNEL)
        vscale_str = parse_header_field(header, 'Vertical Scale')
        h_off_str = parse_header_field(header, 'Vertical Position') or parse_header_field(header, 'Vertical Offset')
        vscale = float(vscale_str) if vscale_str else float('nan')
        host_mean = float(samples.mean()) * (vscale / CODES_PER_DIV)
        meas_str = query_text(':MEASure:MEAN?')
        try:
            meas_mean = float(meas_str)
        except ValueError:
            meas_mean = float('nan')
        diff_mv = (host_mean - meas_mean) * 1000.0
        host_means.append(host_mean)
        meas_means.append(meas_mean)
        print(f'{i:>3}  {host_mean:>14.6f}  {meas_mean:>17.6f}  '
              f'{diff_mv:>+10.3f}  {len(samples):>9d}  {int(samples.min()):>9d}  {int(samples.max()):>9d}  '
              f'{vscale:>8.4f}  {str(h_off_str):>10s}')
    except Exception as e:
        print(f'{i:>3}  ERROR: {e}')
    time.sleep(0.1)

if host_means and meas_means:
    host_arr = np.array(host_means)
    meas_arr = np.array(meas_means)
    print()
    print(f'host  mean over {len(host_arr)} captures: {host_arr.mean():+.6f} V   stdev: {host_arr.std()*1000:.3f} mV')
    print(f'meas  mean over {len(meas_arr)} captures: {meas_arr.mean():+.6f} V   stdev: {meas_arr.std()*1000:.3f} mV')
    print(f'avg(host - meas):                       {(host_arr-meas_arr).mean()*1000:+.3f} mV')
    print(f'stdev(host - meas):                     {(host_arr-meas_arr).std()*1000:.3f} mV')
