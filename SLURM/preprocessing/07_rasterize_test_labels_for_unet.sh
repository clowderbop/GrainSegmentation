INPUT_GPKG="/scratch/s4361687/GrainSeg/dataset/test/test_labels.gpkg"
REFERENCE_TIFF="/scratch/s4361687/GrainSeg/dataset/test/test_PPL.tif"
OUTPUT_RASTER="/scratch/s4361687/GrainSeg/dataset/test/test_labels.tif"

sbatch --export=ALL,INPUT_GPKG="$INPUT_GPKG",REFERENCE_TIFF="$REFERENCE_TIFF",OUTPUT_RASTER="$OUTPUT_RASTER" SLURM/preprocessing/05_rasterize_polygons_for_unet.sh