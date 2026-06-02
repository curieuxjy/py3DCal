"""
NeuralFeels tactile transformer(DPT, dpt_real.p) 를 py3dcal 환경에서 가볍게 로드/추론하기 위한 헬퍼.

- 아키텍처 코드(dpt_model/fusion/head/reassemble)는 neuralfeels 저장소에서 복사한 자체 완결본을 쓴다.
- 추론 전처리/후처리는 neuralfeels TouchVIT.image2heightmap 과 동일하게 맞춘다:
    PIL(RGB) -> Resize(224) -> ToTensor -> Normalize(0.5,0.5) -> model -> depth[0..1]
    -> 원본 크기로 BICUBIC 리사이즈 -> [0..255] 상대 heightmap
- timm 분류 헤드는 사용하지 않으므로 state_dict 는 strict=False 로 로드한다(=timm 버전차에 견고).

dpt_real.p 는 vit_small_patch16_224(.dino) / emb_dim 384 / hooks [2,5,8,11] / resample 128 / patch16 / type=depth.
"""
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .dpt_model import DPTModel

# vit.yaml (scripts/config/main/touch_depth/vit.yaml) 의 General/Dataset 설정과 일치
DPT_CFG = dict(
    image_size=(3, 224, 224),
    patch_size=16,
    emb_dim=384,
    resample_dim=128,
    read="projection",
    hooks=[2, 5, 8, 11],
    nclasses=2,
    type="depth",
    model_timm="vit_small_patch16_224.dino",
)


def build_dpt(weights_path, device="cpu"):
    """DPTModel 생성 + dpt_real.p 가중치 로드(strict=False)."""
    try:
        model = DPTModel(pretrained=False, **DPT_CFG)
    except Exception:
        # 설치된 timm 에 '.dino' 태그가 없으면 베이스 아키텍처로 폴백(가중치는 어차피 로드함)
        cfg = dict(DPT_CFG, model_timm="vit_small_patch16_224")
        model = DPTModel(pretrained=False, **cfg)

    ckpt = torch.load(weights_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # timm 분류 헤드(transformer_encoders.head.*) 정도만 누락/초과면 정상
    interesting = [k for k in (list(missing) + list(unexpected)) if "head." not in k or "head_depth" in k]
    if interesting:
        print(f"[DPT] 주의: 예상치 못한 가중치 키 차이 {len(interesting)}개 (처음 5개): {interesting[:5]}")
    model.eval().to(device)
    return model


_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def dpt_heightmap(model, rgb_image, device="cpu", use_fp16=False, out_size=None):
    """RGB(ndarray, HxWx3) -> DPT heightmap(ndarray, [0..255] 상대값).

    out_size=(W,H) 로 출력 크기를 지정하지 않으면 입력 이미지 크기로 리사이즈한다.
    (neuralfeels TouchVIT.image2heightmap 과 동일한 절차)
    """
    pil = Image.fromarray(rgb_image)
    original_size = pil.size if out_size is None else out_size  # (W, H)
    inp = _TRANSFORM(pil).unsqueeze(0).to(device).float()
    with torch.no_grad():
        if use_fp16 and device == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                out_depth, _ = model(inp)
        else:
            out_depth, _ = model(inp)
    out_depth = out_depth.squeeze(0).float().cpu()           # (1,224,224), [0..1]
    pil_depth = transforms.ToPILImage()(out_depth).resize(original_size, resample=Image.BICUBIC)
    return np.asarray(pil_depth).astype(np.float32)          # [0..255]
