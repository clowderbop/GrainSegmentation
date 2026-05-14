# Tuned watershed instance extraction (loads latest watershed_best_*.json per variant).
sbatch SLURM/unet/run_unet_whole_test_eval.sh \
  --model-dir /scratch/s4361687/GrainSeg/models \
  --image-dir /scratch/s4361687/GrainSeg/dataset/train/cropped \
  --mask-dir /scratch/s4361687/GrainSeg/dataset/train/cropped \
  --output-dir /scratch/s4361687/GrainSeg/eval/watershed_val \
  --watershed-tune-root /scratch/s4361687/GrainSeg/runs/watershed_tune

# Connected-components instance extraction (default --instance-method cc in evaluate.py).
sbatch SLURM/unet/run_unet_whole_test_eval.sh \
  --model-dir /scratch/s4361687/GrainSeg/models \
  --image-dir /scratch/s4361687/GrainSeg/dataset/train/cropped \
  --mask-dir /scratch/s4361687/GrainSeg/dataset/train/cropped \
  --output-dir /scratch/s4361687/GrainSeg/eval/cc_val