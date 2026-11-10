# Executor status

## 当前状态：2026-11-04 07:48 Asia/Shanghai — Phase1 评测完成，待 reporter / 清缓存 / 实验 B

- Phase1 LoRA 30k 训练：TRAIN_EXIT 0；最终 ckpt 29999（另保留 25000）
- Phase1 libero_spatial 评测：496/500（99.2%）；EVAL_DONE @ 07:12 Asia/Shanghai
- GPU 空闲；数据盘约 74%（剩 ~40G）；`/root/autodl-tmp/cache` 约 80G 待清
- 下一授权步骤：reporter 汇报 → 清缓存（保留最终 ckpt）→ 实验 B（只训 action expert LoRA）
