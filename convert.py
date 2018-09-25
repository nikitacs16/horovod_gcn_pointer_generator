import os
import yaml
learning_rate = [0.0004, 0.0001,0.001]
batch_size = [32,64]
base_lstm = 256
stop_count = [17165,8583]
step_count = [345,172]
count = 0
path = 'qbas_baseline'
config = yaml.load(open('config.yaml'))
for lr in learning_rate:
	for k,b in enumerate(batch_size):
		config['exp_name'] = 'qbas_baseline_lr_' + str(lr)+ '_batch_' + str(b) 
		config['stop_steps'] = stop_count[k]
		config['batch_size'] = b
		config['save_steps'] = step_count[k]
		config['adam_lr'] = lr
		config['use_stop_after'] = True
		config['use_save_at'] = True
		config['save_model_seconds'] = 43200
		config['tf_example_format'] = True
		config['word_gcn'] = False
		config['query_gcn'] = False
		config['model_choice'] = 'only_lstm'

		count = count + 1
		file_name = os.path.join(path,'config_' + str(count) + '.yaml')
		yaml.dump(config,open(file_name,'w'))


