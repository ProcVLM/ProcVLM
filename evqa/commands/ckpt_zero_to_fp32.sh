cd /path/to/checkpoint

# python zero_to_fp32.py <checkpoint_dir> <output_file>
python zero_to_fp32.py . /path/to/transformer_model

# after running this script, use copy_config_transformer_fp32.sh
# to copy the config files to the target directory.