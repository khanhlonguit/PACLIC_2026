"""Patch ViCoQA train notebook: nuclear Accelerate device_map fix (TRAIN_FIX_V3)."""
import json
from pathlib import Path

NB = Path(__file__).parent / "train_qwen_lora_unsloth.ipynb"

TRAIN_CELL = r'''def apply_adapter(model, method_name):
    """Gắn đúng PEFT method lên base model đã load bằng Unsloth."""
    from unsloth import FastLanguageModel

    def _resolve_causallm(m):
        if hasattr(m, "prepare_inputs_for_generation"):
            return m
        if hasattr(m, "get_base_model"):
            base = m.get_base_model()
            if hasattr(base, "prepare_inputs_for_generation"):
                return base
        raise RuntimeError("Không tìm thấy ForCausalLM trên model Unsloth.")

    if method_name == "lora":
        model = FastLanguageModel.get_peft_model(
            model,
            r=16, lora_alpha=32, target_modules=TARGET_MODULES,
            lora_dropout=0.05, bias="none",
            use_gradient_checkpointing="unsloth", random_state=3407,
            use_dora=False,
        )
        print("Applied: LoRA (r=16, alpha=32)")
        return model

    if method_name == "dora":
        model = FastLanguageModel.get_peft_model(
            model,
            r=16, lora_alpha=32, target_modules=TARGET_MODULES,
            lora_dropout=0.05, bias="none",
            use_gradient_checkpointing="unsloth", random_state=3407,
            use_dora=True,
        )
        print("Applied: DoRA (r=16, alpha=32, use_dora=True)")
        return model

    if method_name == "tinylora":
        import inspect
        import peft
        from peft import TinyLoraConfig, get_peft_model as peft_get_model

        sig = inspect.signature(TinyLoraConfig.__init__).parameters
        desired = {
            "r": 2, "u": 64, "num_projections": 64,
            "target_modules": TARGET_MODULES,
            "tinylora_dropout": 0.0, "lora_dropout": 0.0,
            "bias": "none", "task_type": "CAUSAL_LM",
            "weight_tying": 0.0, "projection_seed": 3407, "init_weights": True,
        }
        tiny_kwargs = {k: v for k, v in desired.items() if k in sig}
        if "target_modules" not in tiny_kwargs:
            tiny_kwargs["target_modules"] = TARGET_MODULES
        tinylora_config = TinyLoraConfig(**tiny_kwargs)
        causallm = _resolve_causallm(model)
        model = peft_get_model(causallm, tinylora_config)
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.print_trainable_parameters()
        print(f"Applied: TinyLoRA (PEFT {peft.__version__})")
        return model

    if method_name == "delora":
        import peft
        from peft import DeloraConfig, get_peft_model as peft_get_model

        delora_config = DeloraConfig(
            r=16, delora_lambda=15, target_modules=TARGET_MODULES,
            module_dropout=0.05, bias="none", task_type="CAUSAL_LM", init_weights=True,
        )
        causallm = _resolve_causallm(model)
        model = peft_get_model(causallm, delora_config)
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.print_trainable_parameters()
        print(f"Applied: DeLoRA (PEFT {peft.__version__})")
        return model

    raise ValueError(f"Unknown method: {method_name}")


def _force_single_gpu_train_env():
    os.environ["ACCELERATE_BYPASS_DEVICE_MAP"] = "true"
    os.environ["ACCELERATE_NUM_PROCESSES"] = "1"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    for k in ("MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "ROLE_RANK", "ROLE_NAME",
              "LOCAL_WORLD_SIZE", "TORCHELASTIC_RUN_ID", "PET_NNODES"):
        os.environ.pop(k, None)
    try:
        from accelerate.state import AcceleratorState
        AcceleratorState._reset_state(reset_partial_state=True)
    except Exception:
        pass


def _strip_hf_device_map(obj, seen=None):
    if seen is None:
        seen = set()
    if obj is None or id(obj) in seen:
        return
    seen.add(id(obj))
    if hasattr(obj, "hf_device_map"):
        try:
            delattr(obj, "hf_device_map")
        except Exception:
            try:
                obj.hf_device_map = None
            except Exception:
                pass
    for attr in ("model", "base_model", "module", "pretrained_model"):
        child = getattr(obj, attr, None)
        if isinstance(child, torch.nn.Module):
            _strip_hf_device_map(child, seen)


def _patch_accelerate_device_map_check():
    """Bypass cứng verify_device_map — phòng env remote vẫn tưởng distributed."""
    import accelerate.accelerator as acc_mod
    acc_mod.Accelerator.verify_device_map = lambda self, model: False


def train_one_variant(variant, max_seq_length, dataset, eval_dataset=None):
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import SFTTrainer
    from transformers import TrainingArguments, EarlyStoppingCallback
    import inspect
    import sys

    print(">>> TRAIN_FIX_V3: single-GPU / device_map bypass <<<", flush=True)
    _force_single_gpu_train_env()
    _patch_accelerate_device_map_check()
    print(f"  ACCELERATE_BYPASS_DEVICE_MAP={os.environ.get('ACCELERATE_BYPASS_DEVICE_MAP')}", flush=True)

    name = variant["name"]
    print("\n" + "=" * 60)
    print(f" TRAIN VARIANT: {name.upper()}")
    print("=" * 60)

    eval_on = USE_EARLY_STOPPING and eval_dataset is not None
    clear_gpu()

    load_kwargs = dict(
        model_name=BASE_MODEL_NAME,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
    )
    _sig_load = inspect.signature(FastLanguageModel.from_pretrained).parameters
    if "device_map" in _sig_load:
        load_kwargs["device_map"] = {"": 0}

    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    model = apply_adapter(model, name)
    _strip_hf_device_map(model)

    train_args = dict(
        **TRAIN_COMMON,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        output_dir=variant["output_dir"],
        disable_tqdm=False,
        logging_strategy="steps",
        logging_steps=SAVE_STEPS,
        logging_first_step=False,
        log_level="error",
        log_level_replica="error",
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to="none",
        dataloader_num_workers=0,
    )
    _ta_params = inspect.signature(TrainingArguments.__init__).parameters
    _eval_key = "eval_strategy" if "eval_strategy" in _ta_params else "evaluation_strategy"

    callbacks = []
    if eval_on:
        train_args.update({
            _eval_key: "steps",
            "eval_steps": EVAL_STEPS,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
        })
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
        ))
    else:
        train_args[_eval_key] = "no"

    train_args = {k: v for k, v in train_args.items() if k in _ta_params or k == "output_dir"}
    args = TrainingArguments(**train_args)

    sft_kwargs = dict(
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset if eval_on else None,
        args=args,
        callbacks=callbacks,
    )
    _sft_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in _sft_params:
        sft_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in _sft_params:
        sft_kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in _sft_params:
        sft_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in _sft_params:
        sft_kwargs["max_seq_length"] = max_seq_length
    if "packing" in _sft_params:
        sft_kwargs["packing"] = False
    if "dataset_num_proc" in _sft_params:
        sft_kwargs["dataset_num_proc"] = 1

    _force_single_gpu_train_env()
    _patch_accelerate_device_map_check()

    trainer = SFTTrainer(**sft_kwargs)
    if hasattr(trainer, "accelerator"):
        trainer.accelerator.verify_device_map = lambda model: False

    cfg_cls, tr_cls = type(trainer.args), type(trainer)
    sys.modules["trl.trainer.sft_config"] = sys.modules[cfg_cls.__module__]
    sys.modules["trl.trainer.sft_trainer"] = sys.modules[tr_cls.__module__]
    sys.modules[cfg_cls.__module__].SFTConfig = cfg_cls
    sys.modules[tr_cls.__module__].SFTTrainer = tr_cls

    _strip_hf_device_map(trainer.model)
    _force_single_gpu_train_env()
    print(">>> TRAIN_FIX_V3: calling trainer.train() <<<", flush=True)
    trainer.train()

    Path(variant["save_path"]).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(variant["save_path"])
    tokenizer.save_pretrained(variant["save_path"])
    print(f"Saved adapter → {variant['save_path']}")

    del trainer, model, tokenizer
    clear_gpu()
    return variant["save_path"]
'''

LOOP_CELL = r'''# Bắt buộc: env single-GPU TRƯỚC khi train (phòng kernel cũ / remote)
os.environ["ACCELERATE_BYPASS_DEVICE_MAP"] = "true"
os.environ["ACCELERATE_NUM_PROCESSES"] = "1"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
try:
    from accelerate.state import AcceleratorState
    AcceleratorState._reset_state(reset_partial_state=True)
except Exception:
    pass
import accelerate.accelerator as _acc
_acc.Accelerator.verify_device_map = lambda self, model: False
print("Pre-train bypass OK | BYPASS=", os.environ.get("ACCELERATE_BYPASS_DEVICE_MAP"))

if RUN_TRAINING:
    variant_map = {v["name"]: v for v in ADAPTER_VARIANTS}
    trained_paths = {}
    for method in TRAIN_METHODS:
        if method not in variant_map:
            raise ValueError(f"Unknown TRAIN_METHODS item: {method}")
        path = train_one_variant(
            variant_map[method], max_seq_length, dataset,
            eval_dataset=eval_dataset if USE_EARLY_STOPPING else None,
        )
        trained_paths[method] = path
    print("\n✅ Train xong tất cả methods:")
    for k, v in trained_paths.items():
        print(f"  {k}: {v}")
else:
    print("RUN_TRAINING=False — bỏ qua train.")
'''

nb = json.loads(NB.read_text(encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if "def train_one_variant" in src and "def apply_adapter" in src:
        c["source"] = [line + "\n" for line in TRAIN_CELL.splitlines()]
        c["source"][-1] = c["source"][-1].rstrip("\n") + "\n" if c["source"] else []
        # keep trailing newline style as list of lines with \n
        c["source"] = [l + "\n" for l in TRAIN_CELL.split("\n")]
        if c["source"] and c["source"][-1] == "\n":
            c["source"].pop()
            c["source"][-1] = c["source"][-1]  # last line already has \n from join pattern
        # Fix: splitlines drops final empty; rebuild properly
        c["source"] = [line + "\n" for line in TRAIN_CELL.splitlines(True)]
        if c["source"] and not c["source"][-1].endswith("\n"):
            c["source"][-1] += "\n"
        print(f"Patched train cell index {i}")
    if src.strip().startswith("if RUN_TRAINING:") or (
        "if RUN_TRAINING:" in src and "trained_paths" in src and "def train_one_variant" not in src
    ):
        c["source"] = [line + "\n" for line in LOOP_CELL.splitlines(True)]
        if c["source"] and not c["source"][-1].endswith("\n"):
            c["source"][-1] += "\n"
        print(f"Patched loop cell index {i}")

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
print("Done:", NB)
