
.ONESHELL:



%-encoding: batch_size = 10
%-encoding: hidden_layer_dim= 30
%-encoding: hidden_layer_num=1  ## number of hidden layers
%-encoding: subject = 798   # 798
# %-encoding: lags ?= 0 -50 50 #0 50 100 #
%-encoding: lags= $(shell seq -200 100 100) # taking -10s to +2s relative to sentence offset
%-encoding: EPOCHS= 200
%-encoding: train_num= 4000
# %-encoding: taking_words= --taking_words  ##uncomment it to have taking_words==True
%-encoding: activation_function='ReLU'
# %-encoding: second_network= --use_second_network  ##uncomment it to use the second network
%-encoding: learning_rate=0.001
%-encoding: momentum=0.9


%-encoding: num_words= 3  #only valid if taking words==True
%-encoding: min_num_words = 5
%-encoding: electrode = 5
%-encoding: CMD = sbatch --job-name=deep_enc-hidden_layer-$(hidden_layer_num) submit.sh
%-encoding: CMD = python 

# %-srm: JOB_NAME = $(subst /,-,$(desired_fold))
# %-srm: CMD = sbatch --job-name=$(production)-$(JOB_NAME)-across submit.sh


deep-encoding:

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
		--electrode $(min_num_words)\
		--activation_function $(activation_function)\
		--learning_rate $(learning_rate)\
		--momentum $(momentum)\
		$(second_network)\
		$(taking_words)\
		
		
