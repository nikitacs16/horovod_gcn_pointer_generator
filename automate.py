import os
import sys
import yaml
import subprocess
from multiprocessing import Pool

file_name = sys.argv[1]
folds = int(sys.argv[2])
parent_path = sys.argv[3]

def run_train(file_name,out_file_name):
	train_command = 'python run_summarization.py --mode=train --config_file ' + str(file_name) + ' &> out_' + str(out_file_name)
	os.system(train_command)    
	
def run_eval(file_name,out_file_name):
	eval_command = 'python -u run_summarization.py --mode=eval --config_file ' + str(file_name) + ' &> out_v1_' + str(out_file_name)
	os.system('sleep 60; ' + eval_command)

def run_test(file_name,out_file_name):
	test_command = 'python -u run_summarization.py --mode=decode --config_file '+str(file_name) +  ' &> out_t_'+str(out_file_name)
	os.system(test_command)
def run_eval_test(file_name,out_file_name):
	test_command = 'python -u run_summarization.py --mode=decode --use_val_as_test=True --config_file '+str(file_name) +  ' &> out_v_'+str(out_file_name)
	os.system(test_command)


for i in range(1,folds+1):
	with open(file_name) as f:
		doc = yaml.load(f)
	doc['train_path']= os.path.join(parent_path,str(i),'train*.pkl')
	doc['dev_path']= os.path.join(parent_path,str(i),'val*.pkl')
	doc['test_path']= os.path.join(parent_path,str(i),'test*.pkl')
	doc['vocab_path'] =  os.path.join(parent_path,str(i),'vocab')
	doc['exp_name'] = doc['base_experiment'] + '_' + str(i)
	with open(file_name, 'w') as f:
		yaml.dump(doc, f)	

	pool = Pool(processes=2)
	res1 = pool.apply_async(run_train,[file_name,doc['exp_name']])
	res2 = pool.apply_async(run_eval,[file_name,doc['exp_name']])
	pool.close()
	pool.join()
	restore_best_model_command = 'python run_summarization.py --mode=restore_best_model --config_file ' + str(file_name)   
	os.system(restore_best_model)
	pool = Pool(processes=2)
	res1 = pool.apply_async(run_test,[file_name,doc['exp_name']])
	res2 = pool.apply_async(run_eval_test,[file_name,doc['exp_name']])
	pool.close()
	pool.join()
	print(i)



	
print('Completed!')
