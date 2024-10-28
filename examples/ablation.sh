# List of configurations or other parameters for each job
configs=("test", "test_nosolarize", "test_nonorm",  "test_nosolarize_and_norm")  # Add other configurations as needed

# Loop over configurations and submit a job for each one
for config in "${configs[@]}"; do
    python3 train.py --config-name "$config" --config-path ./configs/ \
        hydra/launcher=submitit_slurm \
        hydra.launcher.timeout_min=120 \
        hydra.launcher.cpus_per_task=4 \
        hydra.launcher.gpus_per_task=1 \
        hydra.launcher.partition=gpu -m &
done

wait  # Wait for all jobs to finish