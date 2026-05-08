sbatch SLURM/evaluate_models_and_plot.sh \
  --model-dir /scratch/s4361687/GrainSeg/models \
  --image-dir /scratch/s4361687/GrainSeg/dataset/test/ \
  --mask-dir /scratch/s4361687/GrainSeg/dataset/test/ \
  --output-dir /scratch/s4361687/GrainSeg/eval/unet_test \
  --watershed-tune-root /scratch/s4361687/GrainSeg/runs/watershed_tune
