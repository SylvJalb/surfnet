. scripts/shell_variables.sh 

experiment_name='test'
output_dir='experiments/tracking/'${experiment_name}
create_clean_directory $output_dir

python src/track.py \
    --all_external \
    --data_dir data/external_detections/surfrider_T1_epoch_290_long_segments \
    --output_dir ${output_dir} \
    --confidence_threshold 0.5 \
    --downsampling_factor ${DOWNSAMPLING_FACTOR} \
    --count_threshold 8 \
    --algorithm 'Kalman' \
    --read_from 'folder' \
    --tracker_parameters_dir 'data/tracking_parameters' \
    --output_dir ${output_dir}


