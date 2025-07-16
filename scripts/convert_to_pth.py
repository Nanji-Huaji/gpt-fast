import sys
from pathlib import Path
import torch
import os
from transformers import AutoModelForCausalLM

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import ModelArgs


@torch.inference_mode()
def convert_direct(checkpoint_dir, model_name=None):
    # 确保checkpoint_dir是Path对象
    checkpoint_dir = Path(checkpoint_dir) if isinstance(checkpoint_dir, str) else checkpoint_dir

    if model_name is None:
        model_name = checkpoint_dir.name

    config = ModelArgs.from_name(model_name)
    print(f"Model config {config.__dict__}")

    print(f"Loading model from {checkpoint_dir}...")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)

    # 转换模型权重
    final_result = {}

    # 嵌入层
    final_result["tok_embeddings.weight"] = model.model.embed_tokens.weight

    # 转换每一层
    for i in range(config.n_layer):
        # 注意力层
        q_weight = model.model.layers[i].self_attn.q_proj.weight
        k_weight = model.model.layers[i].self_attn.k_proj.weight
        v_weight = model.model.layers[i].self_attn.v_proj.weight

        # 调整注意力权重形状
        q = permute(q_weight, config.n_head, config)
        k = permute(k_weight, config.n_local_heads, config)

        final_result[f"layers.{i}.attention.wqkv.weight"] = torch.cat([q, k, v_weight])
        final_result[f"layers.{i}.attention.wo.weight"] = model.model.layers[i].self_attn.o_proj.weight

        # FFN层
        final_result[f"layers.{i}.feed_forward.w1.weight"] = model.model.layers[i].mlp.gate_proj.weight
        final_result[f"layers.{i}.feed_forward.w3.weight"] = model.model.layers[i].mlp.up_proj.weight
        final_result[f"layers.{i}.feed_forward.w2.weight"] = model.model.layers[i].mlp.down_proj.weight

        # Norm层
        final_result[f"layers.{i}.attention_norm.weight"] = model.model.layers[i].input_layernorm.weight
        final_result[f"layers.{i}.ffn_norm.weight"] = model.model.layers[i].post_attention_layernorm.weight

    # 最后的Norm和输出层
    final_result["norm.weight"] = model.model.norm.weight
    final_result["output.weight"] = model.lm_head.weight

    # 保存路径
    save_path = checkpoint_dir / "model.pth"
    print(f"Saving checkpoint to {save_path}")
    torch.save(final_result, save_path)


def permute(w, n_head, config):
    dim = config.dim
    return w.view(n_head, 2, config.head_dim // 2, dim).transpose(1, 2).reshape(config.head_dim * n_head, dim)


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description="Convert HuggingFace model directly.")
    # parser.add_argument("--checkpoint_dir", type=str, default="/home/tiantianyi/code/gpt-fast/model/tiny-vicuna-1b")
    # parser.add_argument("--model_name", type=str, default="tiny-vicuna-1b")

    # args = parser.parse_args()
    # convert_direct(args.checkpoint_dir, args.model_name)
    convert_direct("/home/tiantianyi/code/gpt-fast/llama/LLaMa-3.1-13B", "LLaMa-3.1-13B")
