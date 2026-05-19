# nonce-miner-gpu

CUDA GPU miner for **$NONCE** — ERC-8004 Proof-of-Work token on Base mainnet.

Companion to [nonce-miner](https://github.com/citojitu/nonce-miner) (CPU edition). Drop-in compatible — same stdout/stderr protocol, so the same Node.js wrapper that runs the CPU miner can also run this binary instead.

> Expected hashrate on consumer GPUs:
> - **RTX 3060** → ~400-500 MH/s
> - **RTX 4070** → ~800-1000 MH/s
> - **RTX 4090** → ~1.5-2 GH/s
>
> Per-mint time at current difficulty (≈ 6.7e66):
> - RTX 3060: ~40 seconds
> - RTX 4090: ~10 seconds

## Contract

- **Address:** `0xE7bADd12bdf070e925A55A98c981f3aBAB4f20cc` (Base mainnet, chainId 8453)
- **PoW:** `keccak256(challenge || abi.encode(nonce)) < currentDifficulty`
- **Reward:** 100 NONCE per mint (era 0), halving every 100k global mints
- **Block cap:** 10 mints per block

## Requirements

- NVIDIA GPU with CUDA capability 7.5+ (Turing/Ampere/Ada/Hopper)
- CUDA toolkit (nvcc) — install via `apt install nvidia-cuda-toolkit` or [official installer](https://developer.nvidia.com/cuda-downloads)
- Node.js 18+ (for the wrapper script)
- Funded wallet on Base mainnet (~0.001 ETH gas)

## Quick start (rented GPU instance)

Typical workflow for vast.ai / runpod / paperspace:

```bash
# 1. SSH into the GPU instance
ssh user@gpu-instance

# 2. Clone + build
git clone https://github.com/citojitu/Nonce-miner-gpu
cd Nonce-miner-gpu

# Find your GPU's compute capability:
# RTX 30xx (Ampere) = 86, RTX 40xx (Ada) = 89, RTX 50xx (Hopper) = 90
# Default Makefile = 75 (works for most). Override:
make SM=86

# 3. Install Node deps + configure
npm install
cp .env.example .env
nano .env   # paste PRIVATE_KEY

# 4. Run
node miner.js
```

## Build options

```bash
# default (compute_75 = Turing, works on Ampere/Ada too)
make

# explicit compute capability
make SM=86   # Ampere (RTX 30xx, A100)
make SM=89   # Ada    (RTX 40xx)
make SM=90   # Hopper (H100)

# multi-arch fat binary
make ARCH="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89"

# clean
make clean
```

## How it works

```
miner.js (Node)
  └─ fetches state from Base RPC: epoch, difficulty, challenge
  └─ spawns ./nonce-miner-gpu <challenge> <target> <start_nonce>
       └─ CUDA kernel: each GPU thread hashes one nonce, atomic-marks if hit
       └─ outputs "FOUND <nonce>" to stdout when valid
  └─ Node submits mine(nonce) tx via ethers.js
  └─ on epoch flip (every 100 blocks): kill subprocess, re-spawn with new challenge
```

## Architecture

```
nonce-miner-gpu        ← CUDA binary (this repo)
  ├── keccak.cu        ← device-side keccak-256 + comparator
  └── main.cu          ← host program (CUDA memcpy + kernel launch loop)

miner.js               ← Node wrapper (RPC + tx submission)
```

Backend protocol (same as CPU version):
- stdin args: `<challenge_hex_32> <target_hex_32> <start_nonce> [batch_size]`
- stdout: `FOUND <decimal_nonce>` on success
- stderr: `RATE <hashes_per_sec>` every ~2s

## Performance tuning

Default kernel batch = `1 << 22` (4M threads per launch). Adjust via 4th CLI arg:

```bash
./nonce-miner-gpu 0xCHALLENGE 0xTARGET 0 8388608  # 8M threads per launch
```

Larger batch = higher utilization but slower epoch-flip response.

## Security

- **NEVER commit `.env`** — `.gitignore` blocks it
- Use a dedicated miner wallet, fund only what you need
- On rental instances: `.env` lives only on the rented box, deleted on instance teardown

## License

MIT. See [LICENSE](./LICENSE).

## Disclaimer

Not affiliated with the NONCE project. CUDA code may have edge cases — verify with a low-difficulty test target before serious mining. Use at your own risk.
