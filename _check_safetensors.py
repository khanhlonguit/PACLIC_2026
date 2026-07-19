from pathlib import Path
import struct

paths = [
    Path(r'UIT-ViQuAD2.0/qwen2.5-1.5b/LoRa/qwen2.5-1.5b-instruct-delora-viquad2/adapter_model.safetensors'),
    Path(r'ViCoQA/qwen2.5-1.5b/LoRa/qwen2.5-1.5b-instruct-delora-vicoqa/adapter_model.safetensors'),
    Path(r'UIT-ViQuAD2.0/qwen2.5-1.5b/LoRa/qwen2.5-1.5b-instruct-lora-viquad2/adapter_model.safetensors'),
]
for p in paths:
    if not p.exists():
        print(f'MISSING {p}')
        continue
    sz = p.stat().st_size
    head = p.open('rb').read(64)
    print(f'--- {p}')
    print(f'  size={sz} bytes ({sz/1024/1024:.2f} MB)')
    print(f'  head_hex={head[:16].hex()}')
    print(f'  head_ascii={head[:40]!r}')
    if len(head) >= 8:
        hlen = struct.unpack('<Q', head[:8])[0]
        print(f'  claimed_header_len={hlen}')
        if hlen < 100_000_000 and hlen + 8 <= sz:
            hdr = p.open('rb').read(8 + min(hlen, 200))[8:]
            print(f'  header_start={hdr[:120]!r}')
    try:
        from safetensors import safe_open
        with safe_open(str(p), framework='pt', device='cpu') as f:
            keys = list(f.keys())
        print(f'  OK keys={len(keys)} first3={keys[:3]}')
    except Exception as e:
        print(f'  FAIL: {type(e).__name__}: {e}')

d = Path(r'UIT-ViQuAD2.0/qwen2.5-1.5b/LoRa/qwen2.5-1.5b-instruct-delora-viquad2')
print('DIR contents:')
for f in sorted(d.iterdir()):
    print(f'  {f.name} {f.stat().st_size}')
