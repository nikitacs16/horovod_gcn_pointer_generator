import os
import sys
import yaml
file_name = sys.argv[1]
folds = sys.argv[2]
parent_path = sys.argv[3]

for i in range(folds):
	with open(file_name) as f:
        doc = yaml.load(f)
    doc['train_path']= os.path.join(parent_path,str(i),'train*.pkl')
	doc['dev_path']= os.path.join(parent_path,str(i),'val*.pkl')
	doc['test_path']= os.path.join(parent_path,str(i),'test*.pkl')
	doc['vocab_path'] =  os.path.join(parent_path,str(i),'finished_files/vocab')
	doc['exp_name'] = doc['base_name'] + '_' + str(i)
	with open(file_name, 'w') as f:
        yaml.dump(doc, f)	

	os.system('python -u run_summarization.py --mode=train --config ' + str(file_name) + ' &> out')
	os.system('python -u run_summarization.py --mode=restore_best_model --config ' + str(file_name) +' &> out_s')
	os.system('python -u run_summarization.py --mode=decode --config '+str(file_name) +  ' &> out_t' + ' & python -u run_summarization.py --mode=decode --use_val_as_test=True --config '+str(file_name) + ' &> out_v')
	doc['use_stop_after'] = True
	doc['stop_steps'] = 400
	with open(file_name, 'w') as f:
        yaml.dump(doc, f)	
	os.system('python -u run_summarization.py --mode=train --convert_to_coverage_model=True --coverage=True --config '+str(file_name) + ' &> out_c')
	os.system('python -u run_summarization.py --mode=train --coverage=True --config '+str(file_name) + ' &> out')
	os.system('python -u run_summarization.py --mode=decode --coverage=True --config '+str(file_name) +  ' &> out_t' +' & python -u run_summarization.py --coverage=True --use_val_as_test=True --mode=decode --config '+str(file_name) + ' &> out_v')
	print(i)

print('Completed!')
