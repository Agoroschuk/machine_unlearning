import math

# Возможно, логика заморозки будет усложнена
def freeze_transformer_blocks(model, freeze_ratio):
    """
    Замораживает долю самых глубоких transformer-блоков, не замораживая
    самый последний блок.

    Parameters:
    - model: loaded pretrained llm
    - freeze_ratio: fraction of transformer layers to freeze, must be in [0, 1]

    Логика:
    - если всего 48 блоков, замораживаем из первых 47 блоков
    - при 25%: floor(47 * 0.25) = 11 блоков
    - замораживаются самые глубокие из этих 47, то есть блоки перед последним
    - самый последний трансформерный блок всегда остается trainable
    """
    model_type = getattr(model.config, "model_type", "").lower()

    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
        architecture = "gpt2"
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
        architecture = model_type if model_type in {"llama", "phi"} else "llama_or_phi"
    else:
        raise ValueError(
            "Unsupported model architecture. Expected GPT-2, Llama, or Phi layout."
        )

    n_layers = len(blocks)

    if freeze_ratio < 0 or freeze_ratio > 1:
        raise ValueError("freeze_ratio must be in [0, 1]")

    removable_layers = n_layers - 1
    n_freeze = math.floor(removable_layers * freeze_ratio)

    last_block_idx = n_layers - 1
    keep_upto = removable_layers - n_freeze

    frozen_layer_indices = list(range(keep_upto, last_block_idx))

    for layer_idx in frozen_layer_indices:
        for param in blocks[layer_idx].parameters():
            param.requires_grad = False

    model._layer_freeze_metadata = {
        "architecture": architecture,
        "model_type": model_type,
        "original_num_layers": n_layers,
        "new_num_layers": n_layers,
        "freeze_ratio": freeze_ratio,
        "num_frozen_layers": n_freeze,
        "frozen_layer_indices": frozen_layer_indices,
        "kept_last_layer_index": last_block_idx,
    }

    return model
