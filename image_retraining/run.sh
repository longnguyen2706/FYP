#!/bin/bash

echo
echo "<<<<<----- $(date) Running ${0} script file ----->>>>>"

# Setting dir #
HOME_DIR="/home/duclong002/" # provide your home directory
JPEG_DATA_DIR=$HOME_DIR"data_conversion/JPEG_data/" # provide your dataset directory
CODE_DIR=$HOME_DIR"FYP/image_retraining/" # provide where your retrain.py is. If you not refactor the code, please don't change it
#DATASET_NAME=("BreastCancerCell_JPEG" "CHO_JPEG" "Hela_JPEG")
DATASET_NAME=("Hela_JPEG") # provide the dataset name to look for

# print out name of the datasets will be used
echo "The program will do training on these dataset:"
for ix in ${!DATASET_NAME[*]}
do
    printf "    %s \t" "${DATASET_NAME[$ix]}"
done

# Setting hyper param array. The script will loop through each array and do the training with such parameter.
# You should calculate how many cases that will be executed and don't try too large number.
learning_rate_arr=(0.1 0.01 0.001) # set the learning rate value
how_many_training_steps=10000   # set the maximum training steps before terminate
testing_percentage_arr=(10) # set how much percentage of the original data is for testing
validation_percentage_arr=(10)  # set how much percentage of the original data is for validation
train_batch_size_arr=(50 150)   # set the minibatch size
hidden_layer1_size_arr=(50  200) # set numnber of neuron in the hidden layer
dropout_keep_prob_arr=(0.7 0.8) # set the keep prob of the dropout

#learning_rate_arr=(0.1 0.05)
#how_many_training_steps=2000
#testing_percentage_arr=(10)
#validation_percentage_arr=(10)
#train_batch_size_arr=(50)
#hidden_layer1_size_arr=(50)
#dropout_keep_prob_arr=(0.5)

CURRENT_TIME=`date '+%d-%m-%H:%M:%S'`
echo "Current time: $CURRENT_TIME"
log_file=$HOME_DIR"retrain_logs/logfile/log_"$CURRENT_TIME".csv" # create logfile name

# Write to logfile
cat <<EOF >$log_file
Info, Time, Steps, Train accuracy, Validation accuracy, Test accuracy, Final test accuracy,Tensorboard dir, Misclassified Images
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

################# End of Setting ##################

EOF

# Loop through hyper-param arrays
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
						tensorboard_log_file=$HOME_DIR"retrain_logs/"$DATASET"_"`date '+%d-%m-%H:%M:%S'`
                        python3 ${CODE_DIR}retrain.py --image_dir ${image_dir} --how_many_training_steps ${how_many_training_steps} \
                                --learning_rate ${learning_rate} --validation_percentage ${validation_percentage} \
                                --testing_percentage ${testing_percentage} --train_batch_size ${train_batch_size} \
                                --hidden_layer1_size ${hidden_layer1_size} --dropout_keep_prob ${dropout_keep_prob} \
                                --print_misclassified_test_images True --csvlogfile ${log_file} --summaries_dir ${tensorboard_log_file}
                                --architecture 'mobilenet_1.0_224'
					done
				done
			done
		done
	done
done
