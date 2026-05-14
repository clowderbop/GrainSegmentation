sbatch SLURM/test_unet_whole.sh \
  --model-dir /scratch/s4361687/GrainSeg/models \
  --image-dir /scratch/s4361687/GrainSeg/dataset/test/ \
  --mask-dir /scratch/s4361687/GrainSeg/dataset/test/ \
  --output-dir /scratch/s4361687/GrainSeg/eval/unet_test \
  --watershed-tune-root /scratch/s4361687/GrainSeg/runs/watershed_tune
