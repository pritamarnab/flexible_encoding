
.ONESHELL:



%-encoding: batch_size = 10
%-encoding: hidden_layer_dim= 30
%-encoding: hidden_layer_num=1  ## number of hidden layers
%-encoding: subject = 798   # 798
# %-encoding: lags ?= 0 -50 50 #0 50 100 #
%-encoding: lags= $(shell seq -15000 100 100) # taking -10s to +2s relative to sentence offset
%-encoding: EPOCHS= 200
%-encoding: train_num= 0.9 # 90% train, 10% test
# %-encoding: taking_words= --taking_words  ##uncomment it to have taking_words==True
%-encoding: activation_function='ReLU' #
# %-encoding: second_network= --use_second_network  ##uncomment it to use the second network

%-encoding: analysis_level='utterance' # sentence|utterance
%-encoding: max_duration=15 # what is the max utterance duration allowed
%-encoding: model_name_emb='mistral-7b' #'gpt2-xl'| 'mistral-7b' 



%-encoding: learning_rate=0.001
%-encoding: momentum=0.9
%-encoding: all_elec= --all_elec  ##uncomment it to have all_elec==True
%-encoding: num_words= 3  #only valid if taking words==True
%-encoding: min_num_words = 5
%-encoding: electrodes = $(shell seq 1 192)  ##if all_elec==True, the highest electrode can be 192, else 40 
%-encoding: CMD = sbatch --job-name=deep_enc-electrode-$$electrode submit.sh
# %-encoding: CMD = python 
# %-srm: JOB_NAME = $(subst /,-,$(desired_fold))
# %-srm: CMD = sbatch --job-name=$(production)-$(JOB_NAME)-across submit.sh
DT := $(shell date +"%Y%m%d-%H:%M:%S")
a:=$(analysis_level)
b:= $(model_name_emb)

# %-encoding: DIR_NAME := /scratch/gpfs/arnab/flexible_encoding/results/mat_files/mat_files_$(a)_$(b)_$(DT)

deep-encoding:

	@analysis_level=$(analysis_level); \
	model_name=$(model_name_emb); \
	DIR_NAME=/scratch/gpfs/arnab/flexible_encoding/results/mat_files/mat_files_$${analysis_level}_$${model_name_emb}_$(DT); \
		
	mkdir -p $${DIR_NAME}

	save_dir=$${DIR_NAME}; \

	for electrode in $(electrodes); do \

		# # Export DIR_NAME to be used in the submit.sh script
        # export DIR_NAME=$${DIR_NAME}; \

		# $(CMD) /scratch/gpfs/arnab/flexible_encoding/flexible_encoding_arnab_v2.py\
		$(CMD) /scratch/gpfs/arnab/flexible_encoding/scripts/trainer.py\
		--batch_size $(batch_size) \
		--hidden_layer_dim $(hidden_layer_dim) \
		--hidden_layer_num $(hidden_layer_num) \
		--subject $(subject)\
		--lags $(lags)\
		--EPOCHS $(EPOCHS)\
		--train_num $(train_num)\
		--num_words $(num_words)\
		--min_num_words $(min_num_words)\
		--electrode $$electrode\
		--activation_function $(activation_function)\
		--learning_rate $(learning_rate)\
		--momentum $(momentum)\
		--analysis_level $(analysis_level)\
		--max_duration $(max_duration)\
		--save_dir $${save_dir}\
		--model_name_emb $(model_name_emb) \
		$(second_network)\
		$(taking_words)\
		$(all_elec)\
		
	done;



%-pickle: subject = 798   # 798
# %-encoding: lags ?= 0 -50 50 #0 50 100 #
%-pickle: model_name_base_df='gpt2-xl'
%-pickle: model_name_emb='mistral-7b' #'gpt2-xl'| 'mistral-7b' 
%-pickle: min_num_words = 0

%-pickle: CMD = sbatch --job-name=create_pickle submit.sh
# %-pickle: CMD = python 

create-pickle:
	
	$(CMD) /scratch/gpfs/arnab/flexible_encoding/scripts/create_pickle.py\
			--subject $(subject) \
			--min_num_words $(min_num_words) \
			--model_name_base_df $(model_name_base_df) \
			--model_name_emb $(model_name_emb) \


%-plot: subject = 798   # 798
# %-encoding: lags ?= 0 -50 50 #0 50 100 #
%-plot: folder_name='mat_files_utterance__20250312-17:32:13' #folder name where the elec mat files are stored
%-plot: model_name_emb='gpt2-xl' #'gpt2-xl'| 'mistral-7b' 
%-plot: duration = 'medium' # 'short' | 'medium' | 'long'| 'all'

%-plot: CMD = sbatch --job-name=create_plot-duration-$$duration submit.sh
# %-plot: CMD = python 

create-plot:
	
	$(CMD) /scratch/gpfs/arnab/flexible_encoding/scripts/plotting.py\
			--subject $(subject) \
			--folder_name $(folder_name) \
			--model_name_emb $(model_name_emb) \
			--total_duration $(duration) \

clean-empty-dirs:

	find /scratch/gpfs/arnab/flexible_encoding/results/mat_files	-type d	-empty	-delete
	find /scratch/gpfs/arnab/flexible_encoding/results/plots	-type d	-empty	-delete

clean-logs:
	find /scratch/gpfs/arnab/flexible_encoding/logs -type f -delete