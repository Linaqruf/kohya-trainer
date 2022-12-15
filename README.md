# Kohya Trainer V8 - VRAM 12GB
### The Best Way for People Without Good GPUs to Fine-Tune the Stable Diffusion Model

This notebook has been adapted for use in Google Colab based on the [Kohya Guide](https://note.com/kohya_ss/n/nbf7ce8d80f29#c9d7ee61-5779-4436-b4e6-9053741c46bb). </br>
This notebook was adapted by [Linaqruf](https://github.com/Linaqruf)</br>
You can find the latest update to the notebook [here](https://github.com/Linaqruf/kohya-trainer/blob/main/kohya-trainer.ipynb).

## Overview
- Fine tuning of Stable Diffusion's U-Net using Diffusers
- Addressing improvements from the NovelAI article, such as using the output of the penultimate layer of CLIP (Text Encoder) instead of the last layer and learning at non-square resolutions with aspect ratio bucketing.
- Extends token length from 75 to 225 and offers automatic caption and automatic tagging with BLIP, DeepDanbooru, and WD14Tagger
- Supports hypernetwork learning and is compatible with Stable Diffusion v2.0 (base and 768/v)
- By default, does not train Text Encoder for fine tuning of the entire model, but option to train Text Encoder is available.
- Ability to make learning even more flexible than with DreamBooth by preparing a certain number of images (several hundred or more seems to be desirable).

## Run locally 
Please refer to [bmaltais's repo](https://github.com/bmaltais) if you want to run it locally on your terminal
- bmaltais's [kohya_ss](https://github.com/bmaltais/kohya_ss) (dreambooth)
- bmaltais's [kohya_diffusers_fine_tuning](https://github.com/bmaltais/kohya_diffusers_fine_tuning) 
- bmaltais's [kohya_diffusion](https://github.com/bmaltais/kohya_diffusion) (gen_img_diffusers)

## Original post for each dedicated script:
- [gen_img_diffusers](https://note.com/kohya_ss/n/n2693183a798e)
- [merge_vae](https://note.com/kohya_ss/n/nf5893a2e719c)
- [convert_diffusers20_original_sd](https://note.com/kohya_ss/n/n374f316fe4ad)
- [detect_face_rotate](https://note.com/kohya_ss/n/nad3bce9a3622)
- [diffusers_fine_tuning](https://note.com/kohya_ss/n/nbf7ce8d80f29)
- [train_db_fixed](https://note.com/kohya_ss/n/nee3ed1649fb6)
- [merge_block_weighted](https://note.com/kohya_ss/n/n9a485a066d5b)

## Change Logs:

##### v8 (13/12):
- Added support for training with fp16 gradients (experimental feature). This allows training with 8GB VRAM on SD1.x. See "Training with fp16 gradients (experimental feature)" for details.
- Updated WD14Tagger script to automatically download weights.

##### v7 (7/12):
- Requires Diffusers 0.10.2 (0.10.0 or later will work, but there are reported issues with 0.10.0 so we recommend using 0.10.2). To update, run `pip install -U diffusers[torch]==0.10.2` in your virtual environment.
- Added support for Diffusers 0.10 (uses code in Diffusers for `v-parameterization` training and also supports `safetensors`).
- Added support for accelerate 0.15.0.
- Added support for multiple teacher data folders. For caption and tag preprocessing, use the `--full_path` option. The arguments for the cleaning script have also changed, see "Caption and Tag Preprocessing" for details.

##### v6 (6/12):
- Temporary fix for an error when saving in the .safetensors format with some models. If you experienced this error with v5, please try v6.

##### v5 (5/12):
- Added support for the .safetensors format. Install safetensors with `pip install safetensors` and specify the `use_safetensors` option when saving.
- Added the `log_prefix` option.
- The cleaning script can now be used even when one of the captions or tags is missing.

##### v4 (14/12):
- The script name has changed to fine_tune.py.
- Added the option `--train_text_encoder` to train the Text Encoder.
- Added the option `--save_precision` to specify the data format of the saved checkpoint. Can be selected from float, fp16, or bf16.
- Added the option `--save_state` to save the training state, including the optimizer. Can be resumed with the `--resume` option.

##### v3 (29/11):
- Requires Diffusers 0.9.0. To update it, run `pip install -U diffusers[torch]==0.9.0`.
- Supports Stable Diffusion v2.0. Use the `--v2` option when training (and when pre-acquiring latents). If you are using 768-v-ema.ckpt or stable-diffusion-2 instead of stable-diffusion-v2-base, also use the `--v_parameterization` option when training.
- Added options to specify the minimum and maximum resolutions of the bucket when pre-acquiring latents.
- Modified the loss calculation formula.
- Added options for the learning rate scheduler.
- Added support for downloading Diffusers models directly from Hugging Face and for saving during training.
- The cleaning script can now be used even when only one of the captions or tags is missing.
- Added options for the learning rate scheduler.

##### v2 (23/11):
- Implemented Waifu Diffusion 1.4 Tagger for alternative DeepDanbooru for auto-tagging
- Added a tagging script using WD14Tagger.
- Fixed a bug that caused data to be shuffled twice.
- Corrected spelling mistakes in the options for each script.

## Conclusion
> While Stable Diffusion fine tuning is typically based on CompVis, using Diffusers as a base allows for efficient and fast fine tuning with less memory usage. We have also added support for the features proposed by Novel AI, so we hope this article will be useful for those who want to fine tune their models.

 — kohya_ss 

## Credit
[Kohya](https://twitter.com/kohya_ss) | [Lopho](https://github.com/lopho/stable-diffusion-prune) for prune script | Just for my part




