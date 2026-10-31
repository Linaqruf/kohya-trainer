"""The model registry, one row per model family.

To add a model, add a row here, plus a preset if its defaults differ. Tests
check each row against the pinned submodule.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaseModel:
    """A curated base checkpoint a user can pick for a route."""

    key: str
    display_name: str
    repo_id: str
    filename: str | None = None  # None picks the repo's only top-level .safetensors
    download_url: str | None = None  # for files outside the Hub (Civitai, direct links)
    v_prediction: bool = False  # adds the v-pred flags to the config
    # The same model as a ConvRot INT8 file in the same repo. The backend detects it.
    int8_filename: str | None = None
    # Flags that make the backend quantize a bf16 file while it loads.
    # models.prepare_base sets them. An int8 file needs none.
    convrot_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelRoute:
    """One model family: which backend script trains it, and its defaults."""

    key: str
    display_name: str
    backend: str  # "sd-scripts", the only backend today
    entrypoint: str  # the training script at the backend repo root
    preset: str  # the bundled preset TOML under kohya_trainer/presets/
    base_models: tuple[BaseModel, ...]
    default_base: str
    default_vae: BaseModel | None = None
    # A separate text encoder file, and the backend argument that takes it.
    default_text_encoder: BaseModel | None = None
    text_encoder_arg: str = ""
    default_resolution: int = 1024
    # Dataset-config defaults a run can override. A key the dataset config
    # has no place for (caption_dropout_rate) goes to the training settings.
    dataset_defaults: tuple[tuple[str, object], ...] = ()
    # LoCon needs conv layers, which a DiT does not have. The backend's loha and lokr modules
    # raise for SD1.5 (detect_arch_config in networks/network_base.py).
    network_types: tuple[str, ...] = ("lora",)
    # A DiT route's adapter between the text encoder and the DiT. It trains
    # only when a switch in network_args turns it on. adapter_args lists the
    # switch names, the usual one first. adapter_pattern matches its modules,
    # for its own learning rate. adapter_name is what printed notes call it.
    adapter_args: tuple[str, ...] = ()
    adapter_pattern: str = ""
    adapter_name: str = ""
    # The most blocks the trainer can keep in system RAM, or 0 if it cannot swap.
    # Each trainer keeps at least two blocks on the GPU (enable_block_swap in
    # library/anima_models.py and library/moss_image_models.py).
    max_blocks_to_swap: int = 0
    convrot_int8: bool = False  # the trainer takes ConvRot INT8 weights
    # (repo, file) pairs the backend downloads itself. The canary checks them.
    runtime_files: tuple[tuple[str, str], ...] = ()
    gpu: str = "L4"  # the GPU the notebooks ask Colab for
    # Sample preview settings, from each model card. build_plan writes the rest
    # of the built-in prompt. Every negative starts with "nsfw, lowres", because
    # some anime bases drift to nsfw on a bare prompt.
    sample_steps: int = 24
    sample_cfg: float = 6.0
    sample_quality: str = "best quality"
    sample_quality_first: bool = True
    # Write the built-in prompt as a sentence, for a model that reads plain text.
    sample_sentence: bool = False
    default_sample_negative: str = "nsfw, lowres, worst quality, low quality"
    # Images per epoch for the automatic num_repeats (dataset.AUTO_REPEATS_TARGET).
    repeats_target: int = 400
    default_batch_size: int = 2
    # False trains images with no caption file on the trigger word alone.
    default_auto_tag: bool = True
    # False for a model too large to train on a T4 in practice. A T4 run still goes ahead.
    t4_practical: bool = True
    custom_vpred_option: bool = False  # show the custom_base_is_vpred box
    training_note: str = ""  # above the 1-click Training fields
    tuning_note: str = ""  # in the advanced notebook's footer
    about_note: str = ""  # under the title of both notebooks

    def base(self, key: str | None = None) -> BaseModel:
        wanted = key or self.default_base
        for candidate in self.base_models:
            if candidate.key == wanted:
                return candidate
        raise KeyError(
            f"unknown base model {wanted!r} for route {self.key!r}. The known base models are "
            f"{[b.key for b in self.base_models]}."
        )


# The SDXL VAE can decode to black or NaN images in fp16. This one does not.
SDXL_VAE_FP16_FIX = BaseModel(
    key="sdxl-vae-fp16-fix",
    display_name="SDXL VAE (fp16 fix)",
    repo_id="madebyollin/sdxl-vae-fp16-fix",
    filename="sdxl_vae.safetensors",
)

# The improved SD1.5 decoder most people train with.
SD15_VAE_FT_MSE = BaseModel(
    key="vae-ft-mse",
    display_name="SD1.5 VAE (ft-mse)",
    repo_id="stabilityai/sd-vae-ft-mse-original",
    filename="vae-ft-mse-840000-ema-pruned.safetensors",
)

_ROUTES = (
    ModelRoute(
        key="sd15",
        display_name="Stable Diffusion 1.5",
        backend="sd-scripts",
        entrypoint="train_network.py",
        preset="sd15_lora.toml",
        base_models=(
            BaseModel(
                key="anylora",
                display_name="AnyLoRA (neutral LoRA-training base)",
                repo_id="Lykon/AnyLoRA",
                filename="AnyLoRA_bakedVae_blessed_fp16.safetensors",
            ),
            BaseModel(
                key="counterfeit-v3.0",
                display_name="Counterfeit V3.0",
                repo_id="gsdf/Counterfeit-V3.0",
                filename="Counterfeit-V3.0_fp16.safetensors",
            ),
            BaseModel(
                key="sd-1.5",
                display_name="Stable Diffusion 1.5",
                repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
                filename="v1-5-pruned-emaonly.safetensors",
            ),
        ),
        default_base="anylora",
        default_vae=SD15_VAE_FT_MSE,
        default_resolution=512,
        # No text cache here, so tags shuffle. The trigger goes in caption_prefix,
        # which sd-scripts adds before it shuffles.
        dataset_defaults=(("shuffle_caption", True), ("keep_tokens", 1)),
        network_types=("lora", "locon"),
        # The preset's high learning rate with cosine restarts needs fewer steps.
        repeats_target=300,
        sample_cfg=7.0,
        sample_quality="masterpiece, best quality",
        default_sample_negative="nsfw, lowres, worst quality, low quality, normal quality, blurry",
        training_note=(
            "The learning rate, the 3 restarts and about 1,500 steps go together. If you change one, it can overtrain."
        ),
        tuning_note="- `resolution = 768` gives finer detail on anime checkpoints, at a slower step.",
    ),
    ModelRoute(
        key="sdxl",
        display_name="SDXL",
        backend="sd-scripts",
        entrypoint="sdxl_train_network.py",
        preset="sdxl_lora.toml",
        base_models=(
            # Zero is the base published for training. Opt is the tuned release.
            BaseModel(
                key="animagine-xl-4.0-zero",
                display_name="Animagine XL 4.0 Zero (training base)",
                repo_id="cagliostrolab/animagine-xl-4.0-zero",
                filename="animagine-xl-4.0-zero.safetensors",
            ),
            BaseModel(
                key="animagine-xl-4.0-opt",
                display_name="Animagine XL 4.0 Opt",
                repo_id="cagliostrolab/animagine-xl-4.0",
                filename="animagine-xl-4.0-opt.safetensors",
            ),
            BaseModel(
                key="sdxl-base-1.0",
                display_name="Stable Diffusion XL 1.0",
                repo_id="stabilityai/stable-diffusion-xl-base-1.0",
                filename="sd_xl_base_1.0.safetensors",
            ),
        ),
        default_base="animagine-xl-4.0-zero",
        default_vae=SDXL_VAE_FP16_FIX,
        network_types=("lora", "locon", "loha", "lokr"),
        # The Animagine 4.0 card: quality tags last, CFG 5, 28 steps, Euler a.
        sample_steps=28,
        sample_cfg=5.0,
        sample_quality="masterpiece, high score, great score, absurdres",
        sample_quality_first=False,
        default_sample_negative=(
            "nsfw, lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, "
            "cropped, worst quality, low quality, low score, bad score, average score, signature, "
            "watermark, username, blurry"
        ),
        custom_vpred_option=True,
    ),
    ModelRoute(
        key="illustrious",
        display_name="Illustrious / NoobAI",
        backend="sd-scripts",
        entrypoint="sdxl_train_network.py",  # Illustrious is SDXL-architecture
        preset="sdxl_lora.toml",
        base_models=(
            BaseModel(
                key="illustrious-xl-v0.1",
                display_name="Illustrious-XL v0.1",
                repo_id="OnomaAIResearch/Illustrious-xl-early-release-v0",
                filename="Illustrious-XL-v0.1.safetensors",
            ),
            # v1.5 and later moved to 1536 and natural language. Few train on them.
            BaseModel(
                key="illustrious-xl-v1.0",
                display_name="Illustrious-XL v1.0",
                repo_id="OnomaAIResearch/Illustrious-XL-v1.0",
                filename="Illustrious-XL-v1.0.safetensors",
            ),
            BaseModel(
                key="noobai-xl-1.1",
                display_name="NoobAI-XL v1.1 (eps)",
                repo_id="Laxhar/noobai-XL-1.1",
                filename="NoobAI-XL-v1.1.safetensors",
            ),
            BaseModel(
                key="noobai-xl-vpred-1.0",
                display_name="NoobAI-XL v1.0 (v-pred)",
                repo_id="Laxhar/noobai-XL-Vpred-1.0",
                filename="NoobAI-XL-Vpred-v1.0.safetensors",
                v_prediction=True,
            ),
        ),
        default_base="illustrious-xl-v0.1",
        default_vae=SDXL_VAE_FP16_FIX,
        network_types=("lora", "locon", "loha", "lokr"),
        # From the Illustrious v0.1 card, including its example negative.
        sample_quality="masterpiece, best quality",
        sample_quality_first=False,
        default_sample_negative=(
            "nsfw, lowres, worst quality, comic, multiple views, bad quality, low quality, displeasing, "
            "very displeasing, bad anatomy, bad hands, scan artifacts, monochrome, greyscale, "
            "signature, twitter username, jpeg artifacts, 2koma, 4koma, guro, extra digits, fewer digits"
        ),
        tuning_note=(
            "- Leave `noise_offset` at 0, as the Illustrious and NoobAI training bases advise. On the v-pred "
            "base, `zero_terminal_snr` already gives the full range from bright to dark."
        ),
        custom_vpred_option=True,
    ),
    ModelRoute(
        key="anima",
        display_name="Anima",
        backend="sd-scripts",
        entrypoint="anima_train_network.py",
        preset="anima_lora.toml",
        base_models=(
            BaseModel(
                key="anima-base-v1.0",
                display_name="Anima Base v1.0",
                repo_id="circlestone-labs/Anima",
                filename="split_files/diffusion_models/anima-base-v1.0.safetensors",
            ),
        ),
        default_base="anima-base-v1.0",
        default_vae=BaseModel(
            key="qwen-image-vae",
            display_name="Qwen-Image VAE",
            repo_id="circlestone-labs/Anima",
            filename="split_files/vae/qwen_image_vae.safetensors",
        ),
        default_text_encoder=BaseModel(
            key="qwen3-0.6b",
            display_name="Qwen3-0.6B text encoder",
            repo_id="circlestone-labs/Anima",
            filename="split_files/text_encoders/qwen_3_06b_base.safetensors",
        ),
        text_encoder_arg="qwen3",
        network_types=("lora", "loha", "lokr"),  # no LoCon, since the DiT has no Conv2d
        adapter_args=("train_llm_adapter",),
        adapter_pattern=".*llm_adapter.*",
        adapter_name="the LLM adapter",
        max_blocks_to_swap=26,  # 28 blocks
        # The Anima card: 30-50 steps at CFG 4-5, quality and safety tags first.
        # Danbooru art teaches it signature-like scribbles, hence the negative.
        sample_steps=30,
        sample_cfg=5.0,
        sample_quality="masterpiece, best quality, score_7, safe",
        default_sample_negative=(
            "nsfw, lowres, worst quality, low quality, score_1, score_2, score_3, artist name, signature, watermark"
        ),
        tuning_note=(
            "- The upstream doc suggests `network_dim = 8`, `network_alpha = 1` and `learning_rate = 1e-4`: "
            "a quarter of the size, and a little slower to learn than the default 32/32 at 2e-5.\n"
            "- The AdaLN (modulation) layers are left out by default, as upstream does. To train them, add "
            '`"network_args": ["include_patterns=[\'.*_modulation.*\']"]` to `config_overrides`.\n'
            "- On a T4, use `train_batch_size = 1` and a `resolution` of 512 or 768.\n"
            "- If losses go NaN, lower the learning rate first. Then try "
            '`"mixed_precision": "fp16", "full_fp16": True, "no_half_vae": True` in `config_overrides`, and '
            "check the samples closely. bf16 is the reference precision."
        ),
    ),
    ModelRoute(
        key="moss_image",
        display_name="MOSS-Image",
        backend="sd-scripts",
        entrypoint="moss_image_train_network.py",
        preset="moss_image_lora.toml",
        base_models=(
            BaseModel(
                key="moss-image-base",
                display_name="MOSS-Image Base (training base)",
                repo_id="Comfy-Org/MOSS-Image",
                filename="diffusion_models/moss_image_base_bf16.safetensors",
                int8_filename="diffusion_models/moss_image_base_int8_convrot.safetensors",
            ),
            BaseModel(
                key="moss-image-sft",
                display_name="MOSS-Image SFT",
                repo_id="Comfy-Org/MOSS-Image",
                filename="diffusion_models/moss_image_sft_bf16.safetensors",
                int8_filename="diffusion_models/moss_image_sft_int8_convrot.safetensors",
            ),
        ),
        default_base="moss-image-base",
        default_vae=BaseModel(
            key="flux2-vae",
            display_name="FLUX.2 VAE",
            repo_id="Comfy-Org/MOSS-Image",
            filename="vae/flux2-vae.safetensors",
        ),
        # The backend reads the text model and skips the vision tower.
        default_text_encoder=BaseModel(
            key="qwen3.5-9b",
            display_name="Qwen3.5-9B text encoder",
            repo_id="Comfy-Org/MOSS-Image",
            filename="text_encoders/qwen3.5_9b_bf16.safetensors",
            int8_filename="text_encoders/qwen3.5_9b_int8_convrot.safetensors",
        ),
        text_encoder_arg="text_encoder",
        # Without enable_multiline_caption the backend keeps only a caption's
        # first line. The text cache keeps the empty prompt, so dropout works.
        # Dropout helps sampling with an empty negative, as the MOSS pipeline does.
        dataset_defaults=(("enable_multiline_caption", True), ("caption_dropout_rate", 0.1)),
        network_types=("lora", "loha", "lokr"),
        adapter_args=("train_text_embedder", "train_llm_adapter"),
        adapter_pattern=".*text_embedder.*",
        adapter_name="the text embedder",
        max_blocks_to_swap=40,  # 42 blocks
        convrot_int8=True,
        # The tokenizer, which library/moss_image_utils.py loads from this repo.
        runtime_files=(
            ("OpenMOSS-Team/MOSS-Image-Base", "tokenizer/tokenizer.json"),
            ("OpenMOSS-Team/MOSS-Image-Base", "tokenizer/tokenizer_config.json"),
            ("OpenMOSS-Team/MOSS-Image-Base", "tokenizer/chat_template.jinja"),
        ),
        # The official guide: CFG 3 to 6, its negative, sentence prompts. The
        # sampler keeps the official flow shift of 5.0, not the training shift.
        sample_steps=30,
        sample_cfg=4.0,
        sample_quality="best quality",
        sample_sentence=True,
        default_sample_negative="nsfw, lowres, worst quality, bad anatomy, watermark, jpeg artifact",
        # 200 images per epoch at batch 1 over 7 epochs is about 1400 steps.
        default_batch_size=1,
        repeats_target=200,
        # Tag captions name every trait, so the trigger word alone brings out little.
        default_auto_tag=False,
        t4_practical=False,
        training_note=(
            "The learning rate is set for `network_alpha` equal to `network_dim`, so change the three together."
        ),
        about_note=(
            "Most MOSS-Image LoRAs are trained on Base and used on SFT. Base samples look flatter, so judge a "
            "LoRA on SFT. For a character, the trigger word alone teaches the look best, so the tagger is off: "
            "images with no caption file train on the trigger word. To tag them, tick `auto_tag` (`run_tagger` "
            "in the advanced notebook). The 1-click notebook keeps your own caption files as they are."
        ),
        tuning_note=(
            "- The keys below go in `config_overrides`. A dict keeps one `network_args`, so put all its entries "
            "in one list, and all patterns in one `exclude_patterns`.\n"
            "- The default timesteps are MOSS-Image's own pre-training schedule: "
            '`"timestep_sampling": "truncated_logit_normal", "logit_mean": -1.09, "logit_std": 0.95, '
            '"discrete_flow_shift": 1.0`. Keep the minus sign: it puts most steps at high noise. A positive '
            "mean does the reverse. This sampler reads `discrete_flow_shift` too, so keep it at 1.0. "
            'Upstream\'s simpler setting, `"timestep_sampling": "shift"` at `discrete_flow_shift = 3.0`, gives '
            "almost the same noise levels.\n"
            "- LoHa and LoKr usually train at about 2e-4.\n"
            "- If the training outfit shows up in prompts for another outfit, pick an earlier epoch. Or train "
            "again without block 22, which the official sampler skips in the negative half of CFG: "
            '`"network_args": ["exclude_patterns=[\'layers[.]22[.].*\']"]`. This keeps outfit swaps clean in '
            "late epochs, but the trigger word alone can drift more.\n"
            "- On SFT, training the feed-forward layers can break anatomy. To train only attention: "
            '`"network_args": ["exclude_patterns=[\'.*ffn.*\']"]`. With block 22 too: '
            "`\"network_args\": [\"exclude_patterns=['layers[.]22[.].*', '.*ffn.*']\"]`.\n"
            '- To train the text embedder too, add `"network_args": ["train_text_embedder=True"]`. Its rate is '
            "then 5e-5, as upstream advises.\n"
            "- If the GPU runs out of memory, set `convrot_int8` to on in the Base model cell first, then raise "
            "`blocks_to_swap`.\n"
            "- Each caption file is read whole, line breaks included. A `---===---` line splits it into several "
            "captions, and each step picks one.\n"
            "- A LoRA for the text encoder needs much more GPU memory and "
            '`"cache_text_encoder_outputs": None, "cache_text_encoder_outputs_to_disk": None, '
            '"network_train_unet_only": False`.'
        ),
    ),
)

ROUTES: dict[str, ModelRoute] = {route.key: route for route in _ROUTES}


def get(key: str) -> ModelRoute:
    """The route for ``key``. Raises KeyError for an unknown key."""
    try:
        return ROUTES[key]
    except KeyError:
        raise KeyError(f"unknown model route {key!r}. The known routes are {sorted(ROUTES)}.") from None
