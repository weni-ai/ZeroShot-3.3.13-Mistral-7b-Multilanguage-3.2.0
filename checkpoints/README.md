---
library_name: peft
tags:
- trl
- sft
- generated_from_trainer
base_model: mistralai/Mistral-7B-Instruct-v0.2
model-index:
- name: ZeroShot-3.3.13-Mistral-7b-Multilanguage-3.2.0
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ZeroShot-3.3.13-Mistral-7b-Multilanguage-3.2.0

This model is a fine-tuned version of [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3819

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 1
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.4612        | 0.12  | 100  | 0.4510          |
| 0.4184        | 0.25  | 200  | 0.4151          |
| 0.4083        | 0.37  | 300  | 0.4038          |
| 0.393         | 0.5   | 400  | 0.3954          |
| 0.3897        | 0.62  | 500  | 0.3890          |
| 0.3887        | 0.74  | 600  | 0.3847          |
| 0.3825        | 0.87  | 700  | 0.3824          |
| 0.3925        | 0.99  | 800  | 0.3819          |


### Framework versions

- PEFT 0.9.0
- Transformers 4.38.1
- Pytorch 2.1.0+cu121
- Datasets 2.17.1
- Tokenizers 0.15.2