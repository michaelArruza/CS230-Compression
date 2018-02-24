import os

# Directory paths
data_dir = 'dataset/'
pred_dir = "predictions/"
save_dir = 'models/'
simple_data_dir = "simple_dataset/"

# Model-specific names and paths
model = 'Baseline'
experiment = 'simple_1'
model_name = "{}_{}".format(model, experiment)
model_save_dir = os.path.join(save_dir, model)
model_save_path = os.path.join(model_save_dir, model_name)
model_weights_save_path = os.path.join(model_save_dir, model_name + '_weights')
model_predictions_dir = os.path.join(pred_dir, model)
model_predictions_path = os.path.join(model_predictions_dir, model_name)

# Model settings and params
train = True
num_epochs = 50

# Training settings and params
save_model = True # Save model at end of training?
restart = True # If false then it restores the model from the ckpt file specified in save_dir

# Misc:
frame_rate = 44100

def setup():
	# Create directories required
	# Should only run once at beginning!!!!!!
	directories = [data_dir, pred_dir, save_dir, simple_data_dir]
	for directory in directories:
		if not os.path.exists(directory):
			os.makedirs(directory)