# Kohya Trainer
[![GitHub Repo stars](https://img.shields.io/github/stars/Linaqruf/kohya-trainer?style=social)](https://github.com/Linaqruf/kohya-trainer/)</a> [![ko-fi](https://img.shields.io/badge/Support%20me%20on%20Ko--fi-F16061?logo=ko-fi&logoColor=white&style=flat)](https://ko-fi.com/linaqruf) <a href="https://saweria.co/linaqruf"><img alt="Saweria" src="https://img.shields.io/badge/Saweria-7B3F00?style=flat&logo=ko-fi&logoColor=white"/></a>

Github Repository for [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) colab notebook implementation

## Updates
#### 2023
##### v15.0.1 (25/04):
__What Changes?__
- Bump Tensorboard and Tensorflow version to 2.12.0 https://github.com/Linaqruf/kohya-trainer/issues/200
- Fix `class_token` undefined if `activation_word` = 1 word https://github.com/Linaqruf/kohya-trainer/issues/198
    - But also delete `token_to_captions` function, because it's complicated and also it has the same function with `4.2.3. Custom Caption/Tag`
    
##### v15.0.0 (10/04):
__What Changes?__
- Refactoring 4 notebooks
- Use python dict for passing argparse value
- Set AnyLoRA as default pretrained model
- Change AnyLoRA version to bakedVAE
- Added a logic to download multiple custom model or LoRA by separating the link with comma `(,)`
- Revamped `3.3. Image Scraper (Optional)`:
    - Simplified the cell, hide every `parameter` that should be default, e.g. `user_agent`
    - Delete `tags1` and `tags2` field, and changed it to `prompt`. Now user can type their desired tags more than 2 (except: `danbooru`) by separating each tag with a comma `(,)`
    - Added `sub_folder` to scrape images to desired path, useful for multi-concept or multi-directories training.
        - If the value is empty, default path is `train_data_dir`
        - if the value is string and not path, default path is `train_data_dir` + `sub_folder`
        - if the value is path, default path is `sub_folder`
    - Added `range` to limit the number of images to scrape. How to use it: Add `1-200` to download 200 images. Newest Images in the server are prioritized. 
- Added `recursive` option to `4.1. Data Cleaning`, to clean unsupported files and convert RGBA to RGB recursively. Useful for multi-concept or multi-directories training.
- Refactoring `4.2.1. BLIP Captioning`:
    - Added `recursive` option to `4.2.1. BLIP Captioning`, to generate captions recursively, by checking sub-directories as well. Useful for multi-concept or multi-directories training.
    - Set `--debug` or `verbose_logging` in `4.2.1. BLIP Captioning` **On** by default.
- Revamped `Waifu Diffusion 1.4 Tagger V2`:
    - Added WD Tagger new model, and set to default : [SmilingWolf/wd-v1-4-convnextv2-tagger-v2](https://huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2)
    - Added `--remove_underscore` args to the WD Tagger script.
    - Changed how the code works, by not only adding general tags (category = `0`) but also character tags (category = `4`)
    - Character tags can be regulated by specifying `--character_threshold` parameter (default = `0.35`)
    - Changed `--thresh` to `--general_threshold` (default = `0.35`)
    - Added `--undesired_words` args to not add specified words when interrogating images, separate each word by comma, e.g. `1girl, scenery`
    - Changed how `--debug` works, new template :
    ```
        {filename} 
        Character Tags = {character_tags}
        General Tags = {general_tags}
    ```
    - Set `--debug` or `verbose_logging` **On** by default.
    - Added `--frequency_tags` to print tags frequency
    - Added `recursive` option to generate tags recursively, by checking sub-directories as well. Useful for multi-concept or multi-directories training.
- Revamped `4.2.3. Custom Caption/Tag`:
    - Change the code logic by using method from Python List. 
    - Using `append()` to add tags to the end of lines.
    - Using `insert()` to add tags to the beginning of lines.
    - Using `remove()` to remove tags separated by comma.
    - `Cheatsheet` from v14.6 is outdated, now user can easily add or remove tags
        - Tags will be converted to list, so `"_"` or `" "` doesn't matter anymore
        - However, any tags containing `"_"` will still be replaced with `" "`
        - To add a tag, set `1girl` to `custom_tags` then run, if you set `append`, it will be added to the end of lines instead.
        - To add multiple tags, separate each tag by comma `(,)`, e.g. `1girl, garden, school uniform`
        - Note that because of using `insert()`, the result will be backward instead: `school uniform, garden, 1girl`
        - To remove tags, set `custom_tags` to your desired words and set `remove_tags`
    - Added `sub_folder` option, useful for multi-concept or multi-directories training.
        - If the value is `--all` it will process directory and subdirectories in `train_data_dir` recursively.
        - If the value is empty, default path is `train_data_dir`
        - If the value is string and not path, default path is `train_data_dir` + `sub_folder`
        - If the value is path, default path is `sub_folder`
- Finetune notebook:
  - Added `recursive` option to `4.3. Merge Annotation Into JSON` and `Create Buckets & Convert to Latents`
- Revamped `5.2. Dataset Config`:
    - Dreambooth notebook:
        - Deleted `instance_token` and `class_token` and changed into `activation word`
        - Support multi-concept training
            - Recursive, it automatically finds subdirectories, if `supported_extensions` exist `(".png", ".jpg", ".jpeg", ".webp", ".bmp")` it will add the path to `[[dataset.subsets]]` in `dataset_config.toml`
            - You can set parent folder as `train_data_dir` like old version, and you can also normally set `train_data_dir`.
            - To make sure multi-concept training is implemented, I put back folder naming scheme, but now it's optional.
            ```
            <num_repeats>_<class_token>
            ```
            - Example: `10_mikapikazo`, `10` will be added as `num_repeats` and `mikapikazo` will be added to `class_token` in `dataset_config.toml`
            - Because it's optional, if folder naming scheme is not detected, it will get `num_repeats` from `dataset_repeats` and `class_token` from `activation_word`
        - Added `token_to_captions`
            - User can add `activation_word` to captions/tags
            - if folder naming scheme is detected, it will add `<class_token>` from folder name instead of activation word
            - `keep_tokens` set to `1` if `token_to_captions` is enabled
    - Fine-tune notebook:
        - Deleted support for `--dataset_config`, reverted back to old fine-tuning dataset config.
        - Support multi-directory training
            - Set `recursive` to 
                - `4.3. Merge Annotation Into JSON`
                - `4.4. Create Buckets & Convert to Latents`
- Added `min_snr_gamma`, disabled by default, Gamma for reducing the weight of high-loss timesteps. Lower numbers have a stronger effect. The paper recommends 5. Read the paper [here](https://arxiv.org/abs/2303.09556).
- Added `vae_batch_size` to dreambooth notebook
- Revamped `6.4. Launch Portable Web UI` to match the latest [Cagliostro Colab UI](https://github.com/Linaqruf/sd-notebook-collection/blob/main/cagliostro-colab-ui.ipynb)
  - Set `anapnoe-webui` as repo by default.
  
## Useful Links
- Official repository : [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
- Gradio Web UI Implementation : [bmaltais/kohya_ss](https://github.com/bmaltais/kohya_ss)
- Automatic1111 Web UI extensions : [dPn08/kohya-sd-scripts-webui](https://github.com/ddPn08/kohya-sd-scripts-webui)

## Overview
- Fine tuning of Stable Diffusion's U-Net using Diffusers
- Addressing improvements from the NovelAI article, such as using the output of the penultimate layer of CLIP (Text Encoder) instead of the last layer and learning at non-square resolutions with aspect ratio bucketing.
- Extends token length from 75 to 225 and offers automatic caption and automatic tagging with BLIP, DeepDanbooru, and WD14Tagger
- Supports hypernetwork learning and is compatible with Stable Diffusion v2.0 (base and 768/v)
- By default, does not train Text Encoder for fine tuning of the entire model, but option to train Text Encoder is available.
- Ability to make learning even more flexible than with DreamBooth by preparing a certain number of images (several hundred or more seems to be desirable).

## Original post for each dedicated script:
- [gen_img_diffusers](https://note.com/kohya_ss/n/n2693183a798e)
- [merge_vae](https://note.com/kohya_ss/n/nf5893a2e719c)
- [convert_diffusers20_original_sd](https://note.com/kohya_ss/n/n374f316fe4ad)
- [detect_face_rotate](https://note.com/kohya_ss/n/nad3bce9a3622)
- [diffusers_fine_tuning](https://note.com/kohya_ss/n/nbf7ce8d80f29)
- [train_db_fixed](https://note.com/kohya_ss/n/nee3ed1649fb6)
- [merge_block_weighted](https://note.com/kohya_ss/n/n9a485a066d5b)

## Change Logs:

#### 2023
##### v14.6.1 (25/03):
__What Changes?__
- Reformat `1.1. Install Dependencies` cell for all notebooks, added main()
- Downgrade xformers to `0.0.16` and triton to `2.0.0`, because `0.0.17` is now automatically installing `torch 2.0.0` which is incompatible for Colab Notebook, for now. At least no more installing `pre-release` package.
- Fix `libunwind8-dev` not found by installing latest version using `!apt install libunwind8-dev -qq`
- Added condition if `T4` in `!nvidia-smi` output, then do a lowram patch by `sed -i "s@cpu@cuda@" library/model_util.py`
- Added function to `remove_bitsandbytes_message` by manually editing main.py, and then set `os.environ["BITSANDBYTES_NOWELCOME"] = "1"`
  - `BITSANDBYTES_NOWELCOME` is unavailable in `bitsandbytes==0.35.0` and we don't have a plan to update bitsandbytes version
- Deactivate tensorflow print standard error, by `os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"`, credit: [TheLastBen/fast-stable-diffusion](https://github.com/TheLastBen/fast-stable-diffusion)
- Set `LD_LIBRARY_PATH` WITH `os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64/:$LD_LIBRARY_PATH"`

##### v14.6 (21/03):
__What Changes?__
- Reformat 4 main notebook with [black python formatter](https://github.com/psf/black).
- Changed `xformers` wheels version to `0.0.17.dev476` and `triton` to `2.0.0.post1`, to prevent triton installing `torch 2.0.0`.
- Downgrading google perftools and tcmalloc for ram patch, credit to [camenduru/stable-diffusion-webui-colab](https://github.com/camenduru/stable-diffusion-webui-colab).
- Removed `delete metadata` option in `4.1. Data Cleaning` cell, no more accidentally deleted metadata.
- Added `remove_underscore` option in `4.2.2. Waifu Diffusion 1.4 Tagger V2` cell
- Revamped `4.2.3. Custom Caption/Tag (Optional)` cell, now you can add/remove desired tags.
  - Tag example: `1girl, brown hair, school uniform, smile`
  - if your tag has spaces, replace that spaces (` `) with underscore (`_`): 
      - custom_tag	: `blue_berry`
      - output			: `blue berry, 1girl, brown hair, school uniform, smile`
  - if you set `append` to `True`, your custom tag will be added to end of line instead
      - custom_tag	: `blue_berry`
      - output			: `1girl, brown hair, school uniform, smile, blue berry`
  - if you want to add or remove multiple tags, add space (` `):
      - custom_tag : `blue_berry red_juice`, 
      - output: `blue berry, red juice, 1girl, brown hair, school uniform, smile`
  - if you want to remove a tag, set `remove_tag` to `True`
      - custom_tag	: `brown hair`
      - output			: `1girl, school uniform, smile, blue berry`
- Fixes bug when generating `.txt` or `.caption` files in `4.2.3. Custom Caption/Tag (Optional)`, it's added additional (.), e.g. `image..txt`
- Deleted `5.3. Sample prompt config`, sample prompt automatically created. If you want to add another prompt. Edit `sample_prompt.txt` directly in colab notebook editor
- Even though `token` is not `caption`, now you can add token to caption files in `5.2. Dataset Config`, this will function in the same way as the `4.2.3. Custom Caption/Tag (Optional)` cell. By doing this, it automatically set `keep_tokens` to > `1`
  - You can enable or disable the sample prompt in `5.4. Training config`
  - Automatically generating sample every 1 epoch for `LoRA` notebook and every 100 steps for `Dreambooth` and `Finetuning` notebook
  - The prompt weighting such as `( )` and `[ ]` are working.
  - Support long prompt weighting pipeline
- Revamped `5.3. LoRA and Optimizer Config`
  - No more manually setting the `network_module`
  - No more manually setting the `network_args`
  - Added `Recommended Values:
 
    | network_category | network_dim | network_alpha | conv_dim | conv_alpha |
    | :---: | :---: | :---: | :---: | :---: |
    | LoRA | 32 | 1 | - | - |
    | LoCon | 16 | 8 | 8 | 1 |
    | LoHa | 8 | 4 | 4 | 1 |

  - User can choose which `network_category` to train, option: `["LoRA", "LoCon", "LoCon_Lycoris", "LoHa"]`
    - `LoRA` is normal LoRA, only trained cross-attention/transformer layer
    - `LoCon` is LoRA for Convolutional Network but using `networks.lora` as default `network_module`, doesn't support `dropout`
    - `LoCon_Lycoris` is LoRA for Convolutional Network but using `lycoris.kohya` as default `network_module`
      - Why? current state of LoCon trained with lycoris==0.1.3 can't be loaded in Additional Network extension in Web UI, because:
        1. AddNet extension doesn't support `cp_decomposition`
        2. LyCORIS developer is temporarily removing hook support for AddNet extension to prevent code conflict
    - `LoHa` is LoRA with Hadamard Product representation, slower to train than other `network_category`, need more documentation
  - Deleted `network_module` support for `locon.locon_kohya` as it's now deprecated
  - `conv_dim` and `conv_alpha` now has separated markdown field
- Changed `Visualize loss graph (Optional)` position to `6.1`, because it seems has dependency conflict with `6.4. Launch Portable Web UI`
- `6.3. Inference` set default `network_module` to `networks.lora`. Doesn't support LoCon and LoHa trained with LyCORIS.
- Revamped `6.4. Launch Portable Web UI` to match the latest [Cagliostro Colab UI](https://github.com/Linaqruf/sd-notebook-collection/blob/main/cagliostro-colab-ui.ipynb)

##### v14.1 (09/03):
__What Changes?__
- Fix xformers version for all notebook to adapt `Python 3.9.16`
- Added new `network_module` : `lycoris.kohya`. Read [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/Lycoris)
  - Previously LoCon, now it's called `LyCORIS`, a Home for custom network module for [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts).
  - Algo List as of now: 
    - lora: Conventional Methods a.k.a LoCon
    - loha: Hadamard product representation introduced by FedPara
  - For backward compatibility, `locon.locon_kohya` still exist, but you can train LoCon in the new `lycoris.kohya` module as well by specify `["algo=lora"]` in the `network_args`
- Added new condition to enable or disable `generating sample every n epochs/steps`, by disabling it, `sample_every_n_type_value` automatically set to int(999999)

##### v14 (07/03):
__What Changes?__
- Refactoring (again)
  - Moved `support us` button to separated and hidden section
  - Added `old commit` link to all notebook
  - Deleted `clone sd-scripts` option because too risky, small changes my break notebook if new updates contain syntax from python > 3.9 
  - Added `wd-1.5-beta-2` and `wd-1.5-beta-2-aesthetic` as pretrained model for `SDv2.1 768v model`, please use `--v2` and `--v_parameterization` if you wanna train with it.
  - Removed `folder naming scheme` cell for colab dreambooth method, thanks to everyone who made this changes possible. Now you can set `train_data_dir` from gdrive path without worrying `<repeats>_<token> class>` ever again
  -
- Revamped `V. Training Model` section
  - Now it has 6 major cell
    1. Model Config:
        - Specify pretrained model path, vae to use, your project name, outputh path and if you wanna train on `v2` and or `v_parameterization` here.
    2. Dataset Config:
        - This cell will create `dataset_config.toml` file based on your input. And that `.toml` file will be used for training.
        - You can set `class_token` and `num_repeats` here instead of renaming your folder like before.
        - Limitation: even though `--dataset_config` is powerful, but I'm making the workflow to only fit one `train_data_dir` and `reg_data_dir`, so probably it's not compatible to train on multiple concepts anymore. But you can always tweaks `.toml` file.
        - For advanced users, please don't use markdown but instead tweak the python dictionaries yourself, click `show code` and you can add or remove variable, dataset, or dataset.subset from dict, especially if you want to train on multiple concepts.
    3. Sample Prompt Config
        - This cell will create `sample_prompt.txt` file based on your input. And that `.txt` file will be used for generating sample.
        - Specify `sample_every_n_type` if you want to generate sample every n epochs or every n steps.
        - The prompt weighting such as `( )` and `[ ]` are not working.
        - Limitation: Only support 1 line of prompt at a time
        - For advanced users, you can tweak `sample_prompt.txt` and add another prompt based on arguments below.
        - Supported prompt arguments:
            - `--n` : Negative Prompt
            - `--w` : Width
            - `--h` : Height
            - `--d` : Seed, set to -1 for using random seed
            - `--l` : CFG Scale
            - `--s` : Sampler steps
     4. Optimizer Config (LoRA and Optimizer Config)
        - Additional Networks Config:
          - Added support for LoRA in Convolutional Network a.k.a [KohakuBlueleaf/LoCon](https://github.com/KohakuBlueleaf/LoCon) training, please specify `locon.locon_kohya` in `network_module`
          - Revamped `network_args`, now you can specify more than 2 custom args, but you need to specify it inside a list, e.g. `["conv_dim=64","conv_alpha=32"]`
          - `network_args` for LoCon training as follow: `"conv_dim=RANK_FOR_CONV" "conv_alpha=ALPHA_FOR_CONV" "dropout=DROPOUT_RATE"`
          - Remember conv_dim + network_dim, so if you specify both at 128, you probably will get 300mb filesize LoRA
          - Now you can specify if you want to train on both UNet and Text Encoder or just wanna train one of them.
        - Optimizer Config
          - Similar to `network_args`, now you can specify more than 2 custom args, but you need to specify it inside a list, e.g. for DAdaptation : `["decouple=true","weight_decay=0.6"]`
          - Deleted `lr_scheduler_args` and added `lr_scheduler_num_cycles` and `lr_scheduler_power` back
          - Added `Adafactor` for `lr_scheduler`
     5. Training Config
        - This cell will create `config_file.toml` file based on your input. And that `.toml` file will be used for training.
        - Added `num_epochs` back to LoRA notebook and `max_train_steps` to dreambooth and native training 
        - For advanced users, you can tweak training config without re-run specific training cell by editing `config_file.toml`
     6. Start Training
        - Set config path to start training. 
           - sample_prompt.txt
           - config_file.toml
           - dataset_config.toml
        - You can also import training config from other source, but make sure you change all important variable such as what model and what vae did you use 
- Revamped `VI. Testing` section  
  - Deleted all wrong indentation
  - Added `Portable Web UI` as an alternative to try your trained model and LoRA, make sure you still got more time.
- Added new changes to upload `config_file` to huggingface.
##### v13 (25/02):
__What Changes?__
- Of course refactoring, cleaning and make the code and cells more readable and easy to maintain.
  - Moved `Login to Huggingface Hub` to `Deployment` section, in the same cell with defining repo.
  - Merged `Install Kohya Trainer`, `Install Dependencies`, and `Mount Drive` cells
  - Merged `Dataset Cleaning` and `Convert RGB to RGBA` cells
  - Deleted `Image Upscaler` cell, because bucketing automatically upscale your dataset (converted to image latents) to `min_bucket_reso` value.
  - Deleted `Colab Ram Patch` because now you can set `--lowram` in the training script.
  - Revamped `Unzip dataset` cell to make it look simpler
- Added xformers pre-compiled wheel for `A100` 
- Revamped `Pretrained Model` section
  - Deleted some old pretrained model
  - Added `Anything V3.3`, `Chilloutmix`, and `Counterfeit V2.5` as new pretrained model for SD V1.x based model
  - Added `Replicant V1.0`, `WD 1.5 Beta` and `Illuminati Diffusion V1` as new pretrained model for SD V2.x 768v based model
  - Changed `Stable Diffusion 1.5` pretrained model to pruned one.
- Changed Natural Language Captioning back from GIT to BLIP with `beam_search` enabled by default
- Revamped Image Scraper from simple to advanced, added new feature such as:
  - Added `safebooru` to booru list
  - Added `custom_url` option, so you can copy and paste the url instead of specify which booru sites and tags to scrape
  - Added `user_agent` field, because you can't access some image board with default user_agent
  - Added `limit_rate` field to limit your count
  - [Experimental] Added `with_aria2c` to scrape your dataset, not a wrapper, just a simple trick to extract urls with `gallery-dl` and download them with aria2c instead. Fast but seems igonoring `--write-tags`.
  - All downloaded tags now saved with `.txt` format instead of `.jpg.txt`
  - Added `additional_arguments` to make it more flexible if you want to try other args
- Revamped `Append Custom Tag` cell
  - Create new caption file for every image file based on extension provided (`.txt/.caption`) if you didn't want to use BLIP or WD Tagger
  - Added `--keep_tokens` args to the cell
- Revamped `Training Model` section. 
  - Revamped `prettytable` for easier maintenance and bug fixing
  - Now it has 4 major cell:
    - Folder Config
      - To specify `v2`, `v2_parameterization` and all important folder and project_name
    - LoRA and Optimizer Config
      - Only `Optimizer Config` for notebook outside LoRA training
      - All about Optimizer, `learning_rate` and `lr_scheduler` goes here
      - Added new Optimizer from latest kohya-ss/sd-script, all available optimizer : `"AdamW", "AdamW8bit", "Lion", "SGDNesterov", "SGDNesterov8bit", "DAdaptation", "AdaFactor"
      - Currently you can't use `DAdaptation` if you're in Colab free tier because it need more VRAM
      - Added `--optimizer_args` for custom args, useful if you want to try adjusting weight decay, betas etc
    - Dataset Config
      - Only available for Dreambooth method notebook, it basically bucketing cell for Dreambooth.
      - Added `caption dropout`, you can drop your caption or tags by adjusting dropout rates.
      - Added `--bucket_reso_steps` and `--bucket_no_upscale`
    - Training Config
      - Added `--noise_offset`, read [Diffusion With Offset Noise](https://www.crosslabs.org//blog/diffusion-with-offset-noise)
      - Added `--lowram` to load the model in VRAM instead of CPU
- Revamped `Convert Diffusers to Checkpoint` cell, now it's more readable.
- Fixing bugs when `output_dir` located in google drive, it assert an error because of something like `/content/drive/dreambooth_cmd.yaml` which is forbidden, now instead of saved to `{output_dir}`, now training args history are saved to `{training_dir}`

__News__
- I'm in burnout phase, so I'm sorry for the lame update.
- [Fast Kohya Trainer](https://github.com/Linaqruf/kohya-trainer/blob/main/fast-kohya-trainer.ipynb), an idea to merge all Kohya's training script into one cell. Please check it [here](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/fast-kohya-trainer.ipynb). 
  - Please don't expect high, it just a secondary project and maintaining 1-click cell is hard. So I won't prioritized it.
- Kohya Textual Inversion are cancelled for now, because maintaining 4 Colab Notebook already making me this tired. 
  - Please use this instead, not kohya script but everyone on WD server using this since last year:
    - [stable-textual-inversion-cafe colab](https://colab.research.google.com/drive/1bbtGmH0XfQWzVKROhiIP8x5EAv6XuohJ) 
    - [stable-textual-inversion-cafe Colab - Lazy Edition](https://colab.research.google.com/drive/1ouTImTpYkBrX5hiVWrzJeFtOyKES92uV)
- I wrote a Colab Notebook for #AUTOMATIC1111's #stablediffusion Web UI, with built-in Mikubill's #ControlNet extension. All Annotator and extracted ControlNet model are provided in the notebook. It's called [Cagliostro Colab UI](https://colab.research.google.com/github/Linaqruf/sd-notebook-collection/blob/main/cagliostro-colab-ui.ipynb). Please try it.
  - You can use new UI/UX from [Anapnoe](https://github.com/anapnoe/stable-diffusion-webui) in the notebook. You can find the option in `experimental` section.
![image](https://user-images.githubusercontent.com/50163983/221345472-0f9fd9b3-0fe6-4d9d-af0b-05d9e6001e03.png)
 
Training script changes:
- Please read [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) for recent updates.

##### v12 (05/02):
__What Changes?__
- Refactored the 4 notebooks (again)
- Restored the `--learning_rate` function in `kohya-LoRA-dreambooth.ipynb` and `kohya-LoRA-finetuner.ipynb` [#52](https://github.com/Linaqruf/kohya-trainer/issues/52)
- Fixed the cell for inputting custom tags [#48](https://github.com/Linaqruf/kohya-trainer/issues/48) and added the `--keep_tokens` function to prevent custom tags from being shuffled.
- Added a cell to check if all LoRA modules have been trained properly.
- Added descriptions for each notebook and links to the relevant notebooks to prevent "training on the wrong notebook" from happening again.
- Added a cell to check the metadata in the LoRA model.
- Added a cell to change the transparent background in the train data.
- Added a cell to upscale the train data using R-ESRGAN
- Divided the Data Annotation section into two cells:
  - Removed BLIP and replaced it with `Microsoft/GIT` as the auto-captioning for natural language (git-large-textcaps is the default model).
  - Updated the Waifu Diffusion 1.4 Tagger to version v2 (SwinV2 is the default model).
    - The user can adjust the threshold for general tags. It is recommended to set the threshold higher (e.g. `0.85`) if you are training on objects or characters, and lower the threshold (e.g. `0.35`) for training on general, style, or environment.
    - The user can choose from three available models.
- Added a field for uploading to the Huggingface organization account.
- Added the `--min_bucket_reso=320` and `--max_bucket_reso=1280` functions for training resolutions above 512 (e.g. 640 and 768), Thanks Trauter!

Training script Changes([kohya_ss](https://github.com/kohya-ss))
- Please read [Updates 3 Feb. 2023, 2023/2/3](https://github.com/kohya-ss/sd-scripts/blob/fb230aff1b434a21fc679e4902ac1ff5aab1d76b/README.md) for recent updates.

##### v11.5 (31/01):
__What Changes?__
- Refactored the 4 notebooks, removing unhelpful comments and making some code more efficient.
- Removed the `download and generate` regularization images function from `kohya-dreambooth.ipynb` and `kohya-LoRA-dreambooth.ipynb`.
- Simplified cells to create the `train_folder_directory` and `reg_folder_directory` folders in `kohya-dreambooth.ipynb` and `kohya-LoRA-dreambooth.ipynb`.
- Improved the download link function from outside `huggingface` using `aria2c`.
- Set `Anything V3.1` which has been improved CLIP and VAE models as the default pretrained model.
- Fixed the `parameter table` and created the remaining tables for the dreambooth notebooks.
- Added `network_alpha` as a supporting hyperparameter for `network_dim` in the LoRA notebook.
- Added the `lr_scheduler_num_cycles` function for `cosine_with_restarts` and the `lr_scheduler_power` function for `polynomial`.
- Removed the global syntax `--learning_rate` in each LoRA notebook because `unet_lr` and `text_encoder_lr` are already available.
- Fixed the `upload to hf_hub` cell function.

Training script Changes([kohya_ss](https://github.com/kohya-ss))
- Please read [release version 0.4.0](https://github.com/kohya-ss/sd-scripts/releases/tag/v0.4.0) for recent updates.

##### v11 (19/01):
- Reformat notebook, 
  - Added `%store` IPython magic command to store important variable
  - Now you can change the active directory only by editing directory path in `1.1. Clone Kohya Trainer` cell, and save it using `%store` magic command.
  - Deleted `unzip` cell and adjust `download zip` cell to do auto unzip as well if it detect path startswith /content/
  - Added `--flip_aug` to Buckets and Latents cell.
  - Added `--output_name (your-project)` cell to save Trained Model with custom nam.
  - Added ability to auto compress `train_data_dir`, `last-state` and `training_logs` before upload them to Huggingface
- Added `colab_ram_patch` as temporary fix for newest version of Colab after Ubuntu update to `load Stable Diffusion model in GPU instead of RAM`

Training script Changes([kohya_ss](https://github.com/kohya-ss))
- Please read [release version 0.3.0](https://github.com/kohya-ss/sd-scripts/releases/tag/v0.3.0) for recent updates.

##### v10 (02/01) separate release

- Added a function to automatically download the BLIP weight in `make_caption.py`
- Added functions for LoRA training and generation
- Fixed issue where text encoder training was not stopped
- Fixed conversion error for v1 Diffusers->ckpt in `convert_diffusers20_original_sd.py`
- Fixed npz file name for images with dots in `prepare_buckets_latents.py`

Colab UI changes:
- Integrated the repository's format with kohya-ss/sd-script to facilitate merging
- You can no longer choose older script versions in the clone cell because the new format does not support it
- The requirements for both blip and wd tagger have been merged into one requirements.txt file
- The blip cell has been simplified because `make_caption.py` will now automatically download the BLIP weight, as will the wd tagger
- A list of sdv2 models has been added to the "download pretrained model" cell
- The "v2" option has been added to the bucketing and training cells
- An image generation cell using `gen_img_diffusers.py` has been added below the training cell

#### 2022
##### v9 (17/12):
- Added the `save_model_as` option to `fine_tune.py`, which allows you to save the model in any format.
- Added the `keep_tokens` option to `fine_tune.py`, which allows you to fix the first n tokens of the caption and not shuffle them.
- Added support for left-right flipping augmentation in `prepare_buckets_latents.py` and `fine_tune.py` with the `flip_aug` option.

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
