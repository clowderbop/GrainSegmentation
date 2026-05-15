sbatch SLURM/unet/run_unet_whole_test_eval.sh \
  --model-dir /scratch/s4361687/GrainSeg/models \
  --image-dir /scratch/s4361687/GrainSeg/dataset/test/ \
  --mask-dir /scratch/s4361687/GrainSeg/dataset/test/ \
  --gt-gpkg /scratch/s4361687/GrainSeg/dataset/test/test_labels.gpkg \
  --output-dir /scratch/s4361687/GrainSeg/eval/unet_test \
  --watershed-tune-root /scratch/s4361687/GrainSeg/runs/watershed_tune
