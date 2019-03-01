import os
import sys
import yaml
import subprocess

from multiprocessing import Pool

file_name = sys.argv[1]
#gpu_val_id = sys.argv[3]
def run_train(file_name,out_file_name):
	train_command = 'python run_summarization.py --mode=train --config_file ' + str(file_name)
	os.system(train_command)
#	with open(out_file_name, "w") as outfile:
#		subprocess.call(train_command, stdout=outfile)
	
def run_eval(file_name,out_file_name):
	eval_command = 'python  run_summarization.py --mode=eval --config_file ' + str(file_name) 
	eval_command = 'sleep 100; ' + eval_command
	os.system(eval_command)
	#with open(out_file_name, "w") as outfile:
	#	subprocess.call(eval_command, stdout=outfile)

def run_test(file_name,out_file_name):
	test_command = 'python -u run_summarization.py --mode=decode --config_file '+str(file_name) 
	#with open(out_file_name, "w") as outfile:
	#	subprocess.call(test_command, stdout=outfile)
	os.system(test_command)

def run_eval_test(file_name,out_file_name):
	test_command = 'python -u run_summarization.py --mode=decode --use_val_as_test=True --config_file ' + str(file_name)
	#with open(out_file_name, "w") as outfile:
	#	subprocess.call(test_command, stdout=outfile)
	os.system(test_command)	

with open(file_name) as f:
	doc = yaml.load(f)

'''
pool = Pool(processes=2)
res1 = pool.apply_async(run_train,[file_name,doc['exp_name']])
res2 = pool.apply_async(run_eval,[file_name,doc['exp_name']])
pool.close()
pool.join()
'''

restore_best_model_command = 'python run_summarization.py --mode=restore_best_model --config_file ' + str(file_name)
os.system(restore_best_model_command)

pool = Pool(processes=2)
res1 = pool.apply_async(run_test,[file_name,doc['exp_name']])
res2 = pool.apply_async(run_eval_test,[file_name,doc['exp_name']])
pool.close()
pool.join()

#print(i)




#os.system('python  run_summarization.py --mode=train --config_file ' + str(file_name))	

print('Completed!')
