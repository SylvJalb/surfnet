fps=12
segments=short
algorithm='EKF_1'
for filter_type in v0 v1; do 
    for tau in 1 2 3 4 5 6 7 8 9; do 
        files_dir=experiments/tracking/${algorithm}/${segments}_segments/${fps}fps_tau_0
        output_dir=experiments/tracking/${algorithm}/${segments}_segments/${fps}fps_${filter_type}_tau_${tau}
        mkdir ${output_dir}

        cd ${files_dir}

        for f in *.txt; do 
            cd ~/repos/surfnet
            python src/filter_tracks.py \
                --input_file ${files_dir}/$f \
                --filter_type ${filter_type} \
                --min_mean ${tau} \
                --min_len_tracklet ${tau} \
                --output_name ${output_dir}/$f
        done
    done 
done