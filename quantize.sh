# Spits out model at checkpoints/$MODEL_REPO/model_int4.g32.$DEVICE.pth
python quantize.py --checkpoint_path /home/tiantianyi/code/gpt-fast/model/vicuna-68m/model.pth --mode bfloat16

python generate.py --compile --checkpoint_path /home/tiantianyi/code/gpt-fast/model/vicuna-13b-v1.5/model.pth --draft_checkpoint_path /home/tiantianyi/code/gpt-fast/model/tiny-vicuna-1b/model.pth