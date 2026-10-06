#!/usr/bin/env bash
# Stage-2 third-party setup (spec §4), once per machine. Idempotent: re-running skips what is already in place.
#   1. sam2 (facebookresearch, Apache-2.0) at a pinned commit, plus hydra-core 1.3.2 and the rest of its runtime deps,
#      all WITHOUT dependency resolution. sam2's pyproject build-requires torch>=2.5.1, so a plain `pip install` builds
#      in an isolated env that downloads a second torch; SAM2_BUILD_CUDA=0 skips the CUDA extension (only the video
#      predictor's hole filling uses it). The wheel also installs a top-level `training` package (no clash here).
#   2. pysodmetrics 1.6.2 (MIT) without its deps: they pin numpy<2.3.5 and opencv-python-headless, which would downgrade
#      numpy 2.5.2 and write a second cv2 over opencv-python 5.0. Its __init__ imports scikit-image and scikit-learn,
#      which install cleanly against numpy 2.5 (pip dry run: numpy untouched).
#   3. einops and timm (ZoomNeXt's model code imports both; nothing else of ZoomNeXt's requirements is needed for the
#      model). timm is installed WITH dependency resolution: torch / torchvision / huggingface_hub / safetensors are
#      already satisfied, so pip leaves them alone, and the versions guard in step 6 catches any change.
#   4. ZoomNeXt (lartpang, NO licence file: research use only, never committed) cloned at a pinned commit into the
#      git-ignored third_party/zoomnext/. Left pristine: models/seg_models.py works around its CUDA query on CPU.
#   5. checkpoints into weights/pretrained/ (git-ignored): SAM2 v1 Hiera-L (SAM2-UNet starts from v1, not 2.1),
#      SAM2.1 Hiera-L (box-prompted model), ZoomNeXt PVTv2-B2 COD (Google Drive). No ImageNet PVTv2 weights: the COD
#      checkpoint holds the whole encoder.
#   6. verification on CPU: imports, numpy / cv2 / torch / torchvision versions unchanged, every checkpoint strict-loads
#      into its model.
#   bash bash_files/setup_third_party.sh       # foreground, about 2 GB of downloads
# PY is overridable. Afterwards `pip check` reports three unmet pins: pysodmetrics' numpy<2.3.5 and
# opencv-python-headless, and its scikit-image<0.26 (0.26.0 is pinned here; the metrics match py_sod_metrics to 1e-12).
set -euo pipefail
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.." || exit 1
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
SAM2_COMMIT=2b90b9f5ceec907a1c18123530e92e794ad901a4
ZOOMNEXT_COMMIT=614af4348808734aaf2ec43937cf827410330b67
ZOOMNEXT_DRIVE_ID=1_h8XZPDtXMKYUDP2r3MIjLp80eQ1LVBB
W=weights/pretrained
mkdir -p "$W" third_party

versions() { "$PY" -c 'import numpy, cv2, torch, torchvision; print(numpy.__version__, cv2.__version__, torch.__version__, torchvision.__version__)'; }
BEFORE=$(versions)
echo "numpy / cv2 / torch / torchvision before: $BEFORE"

echo "[1/6] sam2 @ $SAM2_COMMIT + hydra-core 1.3.2"
SAM2_BUILD_CUDA=0 "$PY" -m pip install -q --no-build-isolation --no-deps \
    "git+https://github.com/facebookresearch/sam2@$SAM2_COMMIT" \
    hydra-core==1.3.2 omegaconf==2.3.0 antlr4-python3-runtime==4.9.3 iopath==0.1.10 portalocker==4.4.0

echo "[2/6] pysodmetrics 1.6.2 (no deps) + scikit-image 0.26.0 / scikit-learn 1.9.1"
"$PY" -m pip install -q --no-deps pysodmetrics==1.6.2
"$PY" -m pip install -q scikit-image==0.26.0 scikit-learn==1.9.1

echo "[3/6] einops 0.8.1 + timm 1.0.30"
"$PY" -m pip install -q --no-deps einops==0.8.1
"$PY" -m pip install -q timm==1.0.30

echo "[4/6] ZoomNeXt @ $ZOOMNEXT_COMMIT -> third_party/zoomnext (git-ignored, research use only)"
git check-ignore -q third_party/zoomnext/README.md || { echo ".gitignore lacks third_party/zoomnext/: refusing to clone into the repo"; exit 1; }
if [ ! -d third_party/zoomnext/.git ]; then
  git clone -q https://github.com/lartpang/ZoomNeXt third_party/zoomnext
fi
git -C third_party/zoomnext cat-file -e "$ZOOMNEXT_COMMIT^{commit}" 2>/dev/null || git -C third_party/zoomnext fetch -q origin
git -C third_party/zoomnext checkout -q "$ZOOMNEXT_COMMIT"
[ "$(git -C third_party/zoomnext rev-parse HEAD)" = "$ZOOMNEXT_COMMIT" ] || { echo "third_party/zoomnext is not at $ZOOMNEXT_COMMIT"; exit 1; }

echo "[5/6] checkpoints -> $W"
fetch() {   # fetch <url> <dest> <bytes>: kept when present with the right size, else downloaded to .part, size-checked, moved
  local url=$1 dest=$2 size=$3
  if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$size" ]; then echo "  $dest present"; return 0; fi
  curl -L --fail --retry 3 -sS -o "$dest.part" "$url"
  [ "$(stat -c %s "$dest.part")" = "$size" ] || { echo "  $dest: got $(stat -c %s "$dest.part") bytes, expected $size"; exit 1; }
  mv "$dest.part" "$dest"
  echo "  $dest downloaded"
}
fetch https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt "$W/sam2_hiera_large.pt" 897952466
fetch https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt "$W/sam2.1_hiera_large.pt" 898083611
# Google Drive: the usercontent URL with confirm=t skips the virus-scan page; a torch checkpoint is a zip ("PK"), an
# interstitial HTML page would not be (and would have the wrong size anyway)
fetch "https://drive.usercontent.google.com/download?id=$ZOOMNEXT_DRIVE_ID&export=download&confirm=t" "$W/pvtv2-b2-zoomnext.pth" 113020474
[ "$(head -c 2 "$W/pvtv2-b2-zoomnext.pth")" = "PK" ] || { echo "$W/pvtv2-b2-zoomnext.pth is not a torch zip"; exit 1; }

echo "[6/6] verify (CPU)"
AFTER=$(versions)
[ "$AFTER" = "$BEFORE" ] || { echo "numpy / cv2 / torch / torchvision changed: $BEFORE -> $AFTER"; exit 1; }
CUDA_VISIBLE_DEVICES="" "$PY" - <<'EOF'
import sys
import contextlib
import torch
import numpy, cv2, sam2, hydra, py_sod_metrics, skimage, sklearn, einops, timm
from importlib.metadata import version
from sam2.build_sam import build_sam2

print("numpy", numpy.__version__, "| cv2", cv2.__version__, "| torch", torch.__version__, "| SAM-2", version("SAM-2"),
      "| hydra", hydra.__version__, "| pysodmetrics", version("pysodmetrics"), "| skimage", skimage.__version__,
      "| sklearn", sklearn.__version__, "| einops", einops.__version__, "| timm", timm.__version__)
# build_sam2 loads strictly: any missing or unexpected key raises
m = build_sam2("configs/sam2/sam2_hiera_l.yaml", "weights/pretrained/sam2_hiera_large.pt", device="cpu", apply_postprocessing=False)
print("SAM2 v1 Hiera-L ok: trunk channels", m.image_encoder.trunk.channel_list, "patch embed", m.image_encoder.trunk.patch_embed.proj)
del m
m = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", "weights/pretrained/sam2.1_hiera_large.pt", device="cpu",
               apply_postprocessing=False, hydra_overrides_extra=["++model.image_size=512"])
print("SAM2.1 Hiera-L at 512 ok: image embedding", m.sam_image_embedding_size, "x", m.sam_image_embedding_size)
del m


@contextlib.contextmanager
def no_cuda_query():
    # pvt_v2_eff.Attention.__init__ asks torch.cuda for the device capability unconditionally; fake one on CPU
    orig = torch.cuda.get_device_properties
    if not torch.cuda.is_available():
        torch.cuda.get_device_properties = lambda *a, **k: type("P", (), {"major": 0, "minor": 0})()
    try:
        yield
    finally:
        torch.cuda.get_device_properties = orig


sys.path.insert(0, "third_party/zoomnext")
from methods.zoomnext.zoomnext import PvtV2B2_ZoomNeXt
with no_cuda_query():
    net = PvtV2B2_ZoomNeXt(pretrained=False, num_frames=1, input_norm=False, use_checkpoint=False)
ck = torch.load("weights/pretrained/pvtv2-b2-zoomnext.pth", map_location="cpu")
ck = {k: v for k, v in ck.items() if not k.startswith("normalizer.")}
sd = net.state_dict()
extra, missing = [k for k in ck if k not in sd], [k for k in sd if k not in ck]
assert not extra and all(k.endswith("num_batches_tracked") for k in missing), f"unexpected {extra[:5]}, missing {missing[:5]}"
sd.update(ck)
net.load_state_dict(sd, strict=True)
print(f"ZoomNeXt PVTv2-B2 COD ok: {len(ck)} tensors, patch_embed1 {net.encoder.patch_embed1.proj}")
EOF
echo "THIRD-PARTY READY"
