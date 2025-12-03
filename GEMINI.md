🚀 UPGRADED ROBOT BRAIN TRAINING (2025 SOTA)
======================================================================

📋 Configuration:
   Model: 512D, 8 heads, 6 layers
   Action chunks: 48
   Vision: Pretrained VLM (prismatic)
   Diffusion: Flow matching (1-step)

📦 Loading Open X-Embodiment Dataset...

📦 Loading Open X-Embodiment Dataset (train)
   Path: ./open_x_data
   Action chunk size: 48
   Context length: 10
🤗 Loading from HuggingFace...
`trust_remote_code` is not supported anymore.
Please check that the Hugging Face dataset './open_x_data' isn't based on a loading script and remove `trust_remote_code`.
If the dataset is based on a loading script, please ask the dataset author to remove it and convert it to a standard format like Parquet.
⚠️  HuggingFace loading failed: Repo id must use alphanumeric chars, '-', '_' or '.'. The name cannot start or end with '-' or '.' and the maximum length is 96: './open_x_data'.
⚠️  Local loading failed: Data path open_x_data not found
⚠️  Creating dummy dataset for testing
✓ Loaded 1000 episodes
✓ Total samples: 66071


📦 Loading Open X-Embodiment Dataset (val)
   Path: ./open_x_data
   Action chunk size: 48
   Context length: 10
🤗 Loading from HuggingFace...
`trust_remote_code` is not supported anymore.
Please check that the Hugging Face dataset './open_x_data' isn't based on a loading script and remove `trust_remote_code`.
If the dataset is based on a loading script, please ask the dataset author to remove it and convert it to a standard format like Parquet.
⚠️  HuggingFace loading failed: Repo id must use alphanumeric chars, '-', '_' or '.'. The name cannot start or end with '-' or '.' and the maximum length is 96: './open_x_data'.
⚠️  Local loading failed: Data path open_x_data not found
⚠️  Creating dummy dataset for testing
✓ Loaded 100 episodes
✓ Total samples: 7451


🧠 Initializing Robot Brain...
C:\Users\DELL\AppData\Local\Programs\Python\Python313\Lib\site-packages\torch\utils\data\dataloader.py:668: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
