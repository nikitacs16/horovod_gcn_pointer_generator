import os
import yaml
batch_size = [16,32,64]
opt = ['adam','adagrad']
glove = [True,False]
stop_count = [520000,260000,130000]
count = 0
path = '/media/nikita/Q/Sentence/cbt_yaml'
config = yaml.load(open('config _ti_cn.yaml'))
for o in opt:
	for g in glove:
		for k,b in enumerate(batch_size):
			config['exp_name'] = config['base_experiment'] +'_optimizer_' + str(o)+ '_batch_' + str(b) + '_glove_' + str(g)
			config['stop_steps'] = stop_count[k]
			config['batch_size'] = b
			config['optimizer'] = o
			config['use_glove'] = g
			count = count + 1
			file_name = os.path.join(path,'config_' + str(count) + '.yaml')
			yaml.dump(config,open(file_name,'w'))


