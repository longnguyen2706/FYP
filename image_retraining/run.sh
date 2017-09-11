#!/bin/bash
#for x in {1..10}; do (python script.py > /tmp/$x.log ) & done
echo
echo "<<<<<----- $(date) Running ${0} script file ----->>>>>"

# Setting dir #
HOME_DIR="/home/duclong002/"
JPEG_DATA_DIR=$HOME_DIR"Dataset/JPEG_data/"
CODE_DIR=$HOME_DIR"FYP/image_retraining/"
#DATASET_NAME=("BreastCancerCell_JPEG" "CHO_JPEG" "Hela_JPEG")
DATASET_NAME=("Hela_JPEG")

# print out name of the datasets will be used
echo "The program will do training on these dataset:"
for ix in ${!DATASET_NAME[*]}
do
    printf "    %s \t" "${DATASET_NAME[$ix]}"
done

# Setting hyper param array #
learning_rate_arr=(0.1 0.05 0.01 0.005 0.001)
how_many_training_steps=50000
testing_percentage_arr=(10)
validation_percentage_arr=(10)
train_batch_size_arr=(50 100 150)
hidden_layer1_size_arr=(50 100 200)
dropout_keep_prob_arr=(0.5 0.7 0.8)

CURRENT_TIME=`date '+%d-%m-%H:%M'`
echo "Current time: $CURRENT_TIME"
log_file=$HOME_DIR"retrain_logs/logfile/log_"$CURRENT_TIME
cat <<EOF >$log_file
Date created: $(date)

#################### Setting ######################
home_dir: ${HOME_DIR}
Jpeg_data_dir: ${JPEG_DATA_DIR}
code_dir: ${CODE_DIR}
dataset_arr: ${DATASET_NAME[@]}

learning_rate_arr:  ${learning_rate_arr[@]}
how_many_training_steps: ${how_many_training_steps}
testing_percentage_arr: ${testing_percentage_arr[@]}
validation_percentage_arr: ${validation_percentage_arr[@]}
train_batch_size_arr: ${train_batch_size_arr[@]}
hidden_layer1_size_arr: ${hidden_layer1_size_arr[@]}
dropout_keep_prob_arr: ${dropout_keep_prob_arr[@]}
EOF

# Lopp through hyper-param arrays
for DATASET in ${DATASET_NAME[@]}
do 
	image_dir=$JPEG_DATA_DIR$DATASET"/"
	echo "Train dataset dir: $image_dir"
	for learning_rate in ${learning_rate_arr[@]}
	do 
		echo "Learning rate: $learning_rate"
		for testing_percentage_idx in ${!testing_percentage_arr[*]}
		do
			testing_percentage=${testing_percentage_arr[testing_percentage_idx]}
			validation_percentage=${validation_percentage_arr[testing_percentage_idx]}
			echo "Testing percentage: $testing_percentage validation_percentage: $validation_percentage"
			for train_batch_size in ${train_batch_size_arr[@]}
			do
				echo "Train batch size: $train_batch_size"
				for hidden_layer1_size in ${hidden_layer1_size_arr[@]}
				do 
					echo "Hidden layer 1 size: $hidden_layer1_size"
					for dropout_keep_prob in ${dropout_keep_prob_arr[@]}
					do 
						echo "Dropout keep prob: $dropout_keep_prob"


					done
				done
			done
		done
	done
done
#for run in {1..10}
#do
#  command
#done
#
#for f in *
#do
#   i=$(awk 'NR==1{print $1;exit}' $f)
#   myexec $i myfile.extension
#done