#!/usr/bin/env bash
# Create the hsi_camo conda env: Python 3.12 + PyTorch (CUDA 12.8 wheels -- driver 575 supports CUDA <= 12.9, which pins
# torch to 2.11; torch 2.14 is CUDA-13-only) + project deps. Already done on this machine (2026-09-27); kept for reproducibility.
set -euo pipefail
CONDA=${CONDA:-/home/grads/c/chaoyi_he/Desktop/conda}
ENV=$CONDA/envs/hsi_camo
PY=$ENV/bin/python
echo "[1/4] conda create"; "$CONDA/bin/conda" create -y -p "$ENV" python=3.12 >/dev/null
echo "[2/4] pip + torch"; "$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
echo "[3/4] project packages"; "$PY" -m pip install -q numpy scipy h5py pillow pytest ultralytics wandb tensorboard matplotlib pyyaml
echo "[4/4] verify"
"$PY" - <<'EOF'
import torch, torchvision, numpy, scipy, h5py, PIL, ultralytics, wandb, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__, "| cuda available", torch.cuda.is_available(), "| cuda", torch.version.cuda, "| gpus", torch.cuda.device_count())
print("torchvision", torchvision.__version__, "| numpy", numpy.__version__, "| scipy", scipy.__version__, "| h5py", h5py.__version__, "| pillow", PIL.__version__)
print("ultralytics", ultralytics.__version__, "| wandb", wandb.__version__)
x = torch.randn(2, 3, device='cuda'); print("cuda matmul ok:", (x @ x.T).shape)
EOF
echo "ENV READY: $ENV"
