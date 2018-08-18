import os
import yaml
learning_rate = [0.0004, 0.004, 0.04, 0.0001, 0.001, 0.01]
batch_size = [32,64]
base_lstm = [256,128]
stop_count = [17165,8583]
step_count = [345,172]
count = 0
path = '/home/nikita/data/preksha_new/only_query_yaml'
config = yaml.load(open('config.yaml'))
for lr in learning_rate:
	for bl in base_lstm:
		for k,b in enumerate(batch_size):
			config['exp_name'] = 'no_gcn_lr_' + str(lr)+ '_batch_' + str(b) + '_lstm_' + str(bl)
			config['stop_steps'] = step_count[k]
			config['batch_size'] = b
			config['save_steps'] = step_count[k]
			config['adam_lr'] = lr
			count = count + 1
			file_name = os.path.join(path,'config_' + str(count) + '.yaml')
			yaml.dump(config,open(file_name,'w'))


