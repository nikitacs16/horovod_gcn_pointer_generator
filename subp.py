'''
import subprocess
import os

for i in range(2):
	c2 = 'echo hello'

	pc2 = subprocess.Popen(c2, shell=True)
'''
import os
from multiprocessing import Pool
def f1(x,y):
	os.system('echo hi; sleep 2; echo yo')
	print(x)
	print(y)
def f2():
	os.system('echo yes;sleep 3; echo hello')
for i in range(0,2):
	pool = Pool(processes=2)
	res1 = pool.apply_async(f1,[i,i+1])
	res2 = pool.apply_async(f2)
	pool.close()
	pool.join()
	print('Completed!')
