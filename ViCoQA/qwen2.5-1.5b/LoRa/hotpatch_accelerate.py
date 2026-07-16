# PASTE THIS AS A NEW CELL right before the train loop if remote notebook chưa sync.
# Must print: >>> TRAIN_FIX_V3 ...

import os
os.environ["ACCELERATE_BYPASS_DEVICE_MAP"] = "true"
os.environ["ACCELERATE_NUM_PROCESSES"] = "1"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"

try:
    from accelerate.state import AcceleratorState
    AcceleratorState._reset_state(reset_partial_state=True)
except Exception as e:
    print("AcceleratorState reset skip:", e)

import accelerate.accelerator as _acc
_acc.Accelerator.verify_device_map = lambda self, model: False

# Patch in-memory train_one_variant if already defined
if "train_one_variant" in dir():
    _orig = train_one_variant

    def train_one_variant(variant, max_seq_length, dataset, eval_dataset=None):
        print(">>> TRAIN_FIX_V3 (hotpatch) <<<", flush=True)
        os.environ["ACCELERATE_BYPASS_DEVICE_MAP"] = "true"
        try:
            AcceleratorState._reset_state(reset_partial_state=True)
        except Exception:
            pass
        _acc.Accelerator.verify_device_map = lambda self, model: False
        return _orig(variant, max_seq_length, dataset, eval_dataset=eval_dataset)

print("Hotpatch ready | BYPASS=", os.environ.get("ACCELERATE_BYPASS_DEVICE_MAP"))
