# run me with sudo!


import usb.core
import usb.util
import time
import csv
from datetime import datetime
import sys

POLL_INTERVAL = 0.25
DURATION = 2 * 3600
MEASUREMENT = 'MEAN'
OUTFILE = f"scope_log_{datetime.now():%Y%m%d_%H%M%S}.csv"

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
            # Timeout on a single read is fine — keep trying until deadline
            if 'timeout' in str(e).lower():
                continue
            raise
    # print(bytes(buffer))
    return bytes(buffer).decode(errors='replace').strip().rstrip('\x00').rstrip('<')

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
                ch1 = float(scpi_query(f':MEASure:SOURce1 CH1;:MEASure:{MEASUREMENT}?'))
                ch2 = float(scpi_query(f':MEASure:SOURce1 CH2;:MEASure:{MEASUREMENT}?'))
                elapsed = time.time() - start
                w.writerow([f'{elapsed:.3f}', datetime.now().isoformat(), ch1, ch2])
                f.flush()
                sample_count += 1
                
                # Print a status line every 50 samples (5 seconds at 10 Hz)
                if sample_count % 50 == 0:
                    print(f'{elapsed:7.1f}s  CH1={ch1:+.4f}V  CH2={ch2:+.4f}V  ({sample_count} samples)')
            except Exception as e:
                print(f'[{time.time()-start:.1f}s] error: {e}', file=sys.stderr)
            
            sleep_time = POLL_INTERVAL - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print(f'\nStopped at {time.time()-start:.1f}s with {sample_count} samples.')

print(f'\nDone. {sample_count} samples written to {OUTFILE}')