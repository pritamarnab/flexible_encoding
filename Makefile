
.ONESHELL:



%-encoding: batch_size = 10
%-encoding: hidden_layer_dim= 30
%-encoding: hidden_layer_num=1  ## number of hidden layers
%-encoding: subject = 798   # 798
# %-encoding: lags ?= 0 -50 50 #0 50 100 #
%-encoding: lags= $(shell seq -100 100 200) # taking -10s to +2s relative to sentence offset
%-encoding: EPOCHS= 100
%-encoding: train_num= 4000
# %-encoding: taking_words= --taking_words  ##uncomment it to have taking_words==True
%-encoding: activation_function='ReLU'
# %-encoding: second_network= --use_second_network  ##uncomment it to use the second network
%-encoding: learning_rate=0.001
%-encoding: momentum=0.9
%-encoding: all_elec= --all_elec  ##uncomment it to have all_elec==True
%-encoding: num_words= 3  #only valid if taking words==True
%-encoding: min_num_words = 5
%-encoding: electrodes = $(shell seq 0 1)  ##if all_elec==True, the highest electrode can be 192, else 40 
%-encoding: CMD = sbatch --job-name=deep_enc-electrode-$$electrode submit.sh
# %-encoding: CMD = python 
# %-srm: JOB_NAME = $(subst /,-,$(desired_fold))
# %-srm: CMD = sbatch --job-name=$(production)-$(JOB_NAME)-across submit.sh
deep-encoding:
	for electrode in $(electrodes); do \
		$(CMD) /scratch/gpfs/arnab/flexible_encoding/flexible_encoding_arnab_v2.py\
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
		$(second_network)\
		$(taking_words)\
		$(all_elec)\
		
	done;
