# Phase1: add official-compatible LoRA variant of pi05_libero for a single 24GB GPU.
# Same model/data/weights as upstream pi05_libero; only LoRA variants + freeze_filter + no EMA
# (mirrors upstream pi0_libero_low_mem_finetune). Named honestly: pi05_libero_lora.
import pathlib
p = pathlib.Path("/root/autodl-tmp/openpi/src/openpi/training/config.py")
s = p.read_text()
if 'name="pi05_libero_lora"' in s:
    print("already present"); raise SystemExit
anchor = '    #\n    # Fine-tuning Aloha configs.'
block = '''    TrainConfig(
        name="pi05_libero_lora",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=30_000, decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
'''
assert s.count(anchor) == 1
p.write_text(s.replace(anchor, block + anchor))
print("added pi05_libero_lora")
