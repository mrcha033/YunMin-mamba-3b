# Training Architecture

This project trains Mamba-based language models using HuggingFace Accelerate and DeepSpeed. The 3B model
uses 36 layers with a hidden size of 2,560, while the optional 7B variant uses 32 layers and a hidden size
of 4,096. Gradient checkpointing and ZeRO Stage 2 keep GPU memory usage manageable.

Datasets are processed in a curriculum order: `main_data`, `korean_textbook`, `academic_data`, `papers_data`,
`national_data`, `national_assembly_data`, `web_data`, and `social_media_data`. The `train_mamba.py` script
iterates over these categories, saving checkpoints so training can resume if interrupted.

Jobs are launched on Amazon SageMaker using the provided launcher scripts. Spot training jobs can be used to
reduce cost while still resuming from the latest checkpoint when available.
