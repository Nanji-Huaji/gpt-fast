# 启用编译优化，提升推理速度

# 自回归

# python generate.py \
#     --checkpoint_path /home/tiantianyi/code/gpt-fast/llama/llama-3.2-1b/model.pth \
#     --compile \
#     --max_new_tokens 1024 \


# python generate.py \
#     --checkpoint_path /home/tiantianyi/code/gpt-fast/llama/Llama-2-13b-hf/model.pth \
#     --compile \
#     --max_new_tokens 1024 \

# 猜测解码
# 13b 1b
python generate.py --compile --checkpoint_path /home/tiantianyi/code/gpt-fast/llama/Llama-2-13b-hf/model.pth --draft_checkpoint_path //home/tiantianyi/code/gpt-fast/llama/llama-3.2-1b/model.pth

# # 13b 68m
# python generate.py --compile --checkpoint_path /home/tiantianyi/code/gpt-fast/model/vicuna-13b-v1.5/model.pth --draft_checkpoint_path /home/tiantianyi/code/gpt-fast/model/vicuna-68m/model.pth

# # 1b 68m
# python generate.py --compile --checkpoint_path /home/tiantianyi/code/gpt-fast/model/tiny-vicuna-1b/model_bfloat16.pth --draft_checkpoint_path /home/tiantianyi/code/gpt-fast/model/vicuna-68m/model.pth