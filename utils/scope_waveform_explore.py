# run me with sudo!
#
# Pulls raw acquisition memory from CH1 and CH2 and prints the structure
# so we can build a high-precision mean computer that bypasses :MEAS:MEAN?
# rounding. Run once, paste the output back, and we'll iterate the parser.

import usb.core
import usb.util
import time
import struct

CHANNEL = 1  # set to whichever channel currently has the live signal

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


# pyusb >=1.2 has USBTimeoutError; on older versions fall back to string matching.
USBTimeout = getattr(usb.core, 'USBTimeoutError', None)


def _is_timeout(e):
    if USBTimeout is not None and isinstance(e, USBTimeout):
        return True
    s = str(e).lower()
    return 'timed out' in s or 'timeout' in s


def read_all(idle_ms=400, total_timeout_s=8.0):
    """Read until idle for idle_ms or total_timeout_s elapses. Handles multi-packet binary."""
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


def query(cmd, **kw):
    ep_out.write(cmd.encode() + b'\n')
    return read_all(**kw)


def query_text(cmd):
    return query(cmd, idle_ms=200, total_timeout_s=2.0).decode(errors='replace').strip().rstrip('\x00').rstrip('<')


print("=" * 70)
print("IDN:", query_text('*IDN?'))
print("=" * 70)

# 1) Scaling info we'd need to convert ADC codes -> volts.
print("\n--- Scaling queries ---")
for q in [
    f':CHANnel{CHANNEL}:SCALe?',
    f':CHANnel{CHANNEL}:OFFSet?',
    f':CHANnel{CHANNEL}:POSition?',
    f':CHANnel{CHANNEL}:PROBe:RATio?',
    f':CHANnel{CHANNEL}:COUPling?',
    f':TIMebase:SCALe?',
    f':ACQuire{CHANNEL}:LENGth?',
    f':MEASure:SOURce1 CH{CHANNEL};:MEASure:MEAN?',
]:
    try:
        ans = query_text(q)
    except Exception as e:
        ans = f'<error: {e}>'
    print(f'  {q:60s} -> {ans!r}')

# 2) Pull the raw memory.
print(f"\n--- Pulling :ACQuire{CHANNEL}:MEMory? ---")
raw = query(f':ACQuire{CHANNEL}:MEMory?', idle_ms=500, total_timeout_s=10.0)
print(f"Total reply length: {len(raw)} bytes")

# 3) Show the ASCII portion (header) and locate the IEEE-488.2 block marker (#).
hash_idx = raw.find(b'#')
print(f"First '#' at byte {hash_idx}")
if hash_idx > 0:
    ascii_header = raw[:hash_idx].decode(errors='replace')
    print("\n--- ASCII header (everything before first '#') ---")
    print(ascii_header)
    # Standard IEEE-488.2: '#' then one digit N, then N digits = byte count, then binary.
    n_digits = int(chr(raw[hash_idx + 1]))
    byte_count = int(raw[hash_idx + 2:hash_idx + 2 + n_digits].decode())
    data_start = hash_idx + 2 + n_digits
    data_end = data_start + byte_count
    print(f"\n--- Block header ---")
    print(f"  N-digits: {n_digits}")
    print(f"  Declared byte count: {byte_count}")
    print(f"  Data slice: bytes [{data_start}:{data_end}]  (file has {len(raw)})")

    data = raw[data_start:data_end]
    print(f"  Got {len(data)} data bytes")
    print(f"  First 24 bytes (hex): {data[:24].hex()}")
    print(f"  Last 8 bytes (hex):   {data[-8:].hex() if len(data) >= 8 else data.hex()}")

    # 4) Try interpreting as int16 big-endian (most GDS firmware) and little-endian.
    if byte_count > 0 and byte_count % 2 == 0:
        n_samples = byte_count // 2
        be = struct.unpack(f'>{n_samples}h', data[:byte_count])
        le = struct.unpack(f'<{n_samples}h', data[:byte_count])
        print(f"\n--- int16 sample stats ---")
        print(f"  count={n_samples}")
        print(f"  big-endian:    min={min(be):6d}  max={max(be):6d}  mean={sum(be)/n_samples:+9.3f}")
        print(f"  little-endian: min={min(le):6d}  max={max(le):6d}  mean={sum(le)/n_samples:+9.3f}")

        # Heuristic: pick the orientation whose range is reasonable for an 8-bit-ish ADC.
        # GDS scopes typically map full screen to roughly +/- 12500 codes (25 codes/div * 10 divs / 2)
        # but firmware varies; we'll show both and let the human pick.
else:
    print("No '#' marker found — reply may be all ASCII or truncated.")
    print(f"First 200 bytes: {raw[:200]!r}")

print("\nDone. Paste this whole block back so we can finalize the parser.")
