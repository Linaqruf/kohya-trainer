# License of this file is ASL 2.0

import argparse
import torch


VAE_PREFIX = "first_stage_model."

# copy from convert_diffusers_to_original_stable_diffusion.py ASL 2.0

# ================#
# VAE Conversion #
# ================#

vae_conversion_map = [
    # (stable-diffusion, HF Diffusers)
    ("nin_shortcut", "conv_shortcut"),
    ("norm_out", "conv_norm_out"),
    ("mid.attn_1.", "mid_block.attentions.0."),
]

for i in range(4):
  # down_blocks have two resnets
  for j in range(2):
    hf_down_prefix = f"encoder.down_blocks.{i}.resnets.{j}."
    sd_down_prefix = f"encoder.down.{i}.block.{j}."
    vae_conversion_map.append((sd_down_prefix, hf_down_prefix))

  if i < 3:
    hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0."
    sd_downsample_prefix = f"down.{i}.downsample."
    vae_conversion_map.append((sd_downsample_prefix, hf_downsample_prefix))

    hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
    sd_upsample_prefix = f"up.{3-i}.upsample."
    vae_conversion_map.append((sd_upsample_prefix, hf_upsample_prefix))

  # up_blocks have three resnets
  # also, up blocks in hf are numbered in reverse from sd
  for j in range(3):
    hf_up_prefix = f"decoder.up_blocks.{i}.resnets.{j}."
    sd_up_prefix = f"decoder.up.{3-i}.block.{j}."
    vae_conversion_map.append((sd_up_prefix, hf_up_prefix))

# this part accounts for mid blocks in both the encoder and the decoder
for i in range(2):
  hf_mid_res_prefix = f"mid_block.resnets.{i}."
  sd_mid_res_prefix = f"mid.block_{i+1}."
  vae_conversion_map.append((sd_mid_res_prefix, hf_mid_res_prefix))


vae_conversion_map_attn = [
    # (stable-diffusion, HF Diffusers)
    ("norm.", "group_norm."),
    ("q.", "query."),
    ("k.", "key."),
    ("v.", "value."),
    ("proj_out.", "proj_attn."),
]


def reshape_weight_for_sd(w):
  # convert HF linear weights to SD conv2d weights
  return w.reshape(*w.shape, 1, 1)


def convert_vae_state_dict(vae_state_dict):
  mapping = {k: k for k in vae_state_dict.keys()}
  for k, v in mapping.items():
    for sd_part, hf_part in vae_conversion_map:
      v = v.replace(hf_part, sd_part)
    mapping[k] = v
  for k, v in mapping.items():
    if "attentions" in k:
      for sd_part, hf_part in vae_conversion_map_attn:
        v = v.replace(hf_part, sd_part)
      mapping[k] = v
  new_state_dict = {v: vae_state_dict[k] for k, v in mapping.items()}
  weights_to_convert = ["q", "k", "v", "proj_out"]
  for k, v in new_state_dict.items():
    for weight_name in weights_to_convert:
      if f"mid.attn_1.{weight_name}.weight" in k:
        print(f"Reshaping {k} for SD format")
        new_state_dict[k] = reshape_weight_for_sd(v)
  return new_state_dict


def convert_diffusers_vae(vae_path):
  vae_state_dict = torch.load(vae_path, map_location="cpu")
  vae_state_dict = convert_vae_state_dict(vae_state_dict)
  return vae_state_dict


def merge_vae(ckpt, vae, output):
  print(f"load checkpoint: {ckpt}")
  model = torch.load(ckpt, map_location="cpu")
  sd = model['state_dict']

  full_model = False

  print(f"load VAE: {vae}")
  if vae.endswith(".bin"):
    print("convert diffusers VAE to stablediffusion")
    vae_sd = convert_diffusers_vae(vae)
  else:
    vae_model = torch.load(vae, map_location="cpu")
    vae_sd = vae_model['state_dict']

    # vae only or full model
    for vae_key in vae_sd:
      if vae_key.startswith(VAE_PREFIX):
        full_model = True
        break

  count = 0
  for vae_key in vae_sd:
    sd_key = vae_key
    if full_model:
      if not sd_key.startswith(VAE_PREFIX):
        continue
    else:
      if sd_key not in sd:
        sd_key = VAE_PREFIX + sd_key
    if sd_key not in sd:
      print(f"key not exists in model: {vae_key}")
      continue
    sd[sd_key] = vae_sd[vae_key]
    count += 1
  print(f"{count} weights are copied")

  print(f"saving checkpoint to: {output}")
  torch.save(model, output)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("ckpt", type=str, help="target checkpoint to replace VAE / マージ対象のモデルcheckpoint")
  parser.add_argument("vae", type=str, help="VAE/model checkpoint to merge / マージするVAEまたはモデルのcheckpoint")
  parser.add_argument("output", type=str, help="output checkoint / 出力先checkpoint")
  args = parser.parse_args()

  merge_vae(args.ckpt, args.vae, args.output)
