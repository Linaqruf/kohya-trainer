# original code: https://github.com/eyriewow/merge-models

import os
import argparse
import re
import torch
from tqdm import tqdm


NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS


def merge(args):
  if args.weights is None:
    weights = None
  else:
    weights = [float(w) for w in args.weights.split(',')]
    if len(weights) != NUM_TOTAL_BLOCKS:
      print(f"weights value must be {NUM_TOTAL_BLOCKS}.")
      return

  device = args.device
  print("loading", args.model_0)
  model_0 = torch.load(args.model_0, map_location=device)
  print("loading", args.model_1)
  model_1 = torch.load(args.model_1, map_location=device)
  theta_0 = model_0["state_dict"]
  theta_1 = model_1["state_dict"]
  alpha = args.base_alpha

  output_file = f'{args.output}-{str(alpha)[2:] + "0"}-bw.ckpt'

  # check if output file already exists, ask to overwrite
  if os.path.isfile(output_file):
    print("Output file already exists. Overwrite? (y/n)")
    while True:
      overwrite = input()
      if overwrite == "y":
        break
      elif overwrite == "n":
        print("Exiting...")
        return
      else:
        print("Please enter y or n")

  re_inp = re.compile(r'\.input_blocks\.(\d+)\.')           # 12
  re_mid = re.compile(r'\.middle_block\.(\d+)\.')           # 1
  re_out = re.compile(r'\.output_blocks\.(\d+)\.')           # 12

  for key in (tqdm(theta_0.keys(), desc="Stage 1/2") if not args.verbose else theta_0.keys()):
    if "model" in key and key in theta_1:
      current_alpha = alpha

      # check weighted and U-Net or not
      if weights is not None and 'model.diffusion_model.' in key:
        # check block index
        weight_index = -1

        if 'time_embed' in key:
          weight_index = 0                # before input blocks
        elif '.out.' in key:
          weight_index = NUM_TOTAL_BLOCKS - 1     # after output blocks
        else:
          m = re_inp.search(key)
          if m:
            inp_idx = int(m.groups()[0])
            weight_index = inp_idx
          else:
            m = re_mid.search(key)
            if m:
              weight_index = NUM_INPUT_BLOCKS
            else:
              m = re_out.search(key)
              if m:
                out_idx = int(m.groups()[0])
                weight_index = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + out_idx

        if weight_index >= NUM_TOTAL_BLOCKS:
          print(f"error. illegal block index: {key}")
        if weight_index >= 0:
          current_alpha = weights[weight_index]
          if args.verbose:
            print(f"weighted '{key}': {current_alpha}")

      theta_0[key] = (1 - current_alpha) * theta_0[key] + current_alpha * theta_1[key]

  for key in tqdm(theta_1.keys(), desc="Stage 2/2"):
    if "model" in key and key not in theta_0:
      theta_0[key] = theta_1[key]

  print("Saving...")

  torch.save({"state_dict": theta_0}, output_file)

  print("Done!")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Merge two models with weights for each block")
  parser.add_argument("model_0", type=str, help="Path to model 0")
  parser.add_argument("model_1", type=str, help="Path to model 1")
  parser.add_argument("--base_alpha", type=float,
                      help="Alpha value (for model 0) except U-Net, optional, defaults to 0.5", default=0.5, required=False)
  parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)
  parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)
  parser.add_argument("--weights", type=str,
                      help=f"comma separated {NUM_TOTAL_BLOCKS} weights value (for model 0) for each U-Net block", default=None, required=False)
  parser.add_argument("--verbose", action='store_true', help="show each block weight", required=False)

  args = parser.parse_args()
  merge(args)
