from __future__ import print_function
import sys
import glob
import os

input_path = sys.argv[1]

dec = open(os.path.join(input_path, 'decoded_n.txt'),'w')
ref = open(os.path.join(input_path,  'reference_n.txt'), 'w')

dec_files = glob.glob(os.path.join(input_path, 'decoded/*.txt'))
ref_files = glob.glob(os.path.join(input_path, 'reference/*.txt'))

for i in sorted(dec_files):
	f = open(i,'r')
	s = ''
	for k in f.readlines():
		s = s + k.strip().replace('\n','')
	#print(s)
	print(s,file=dec)

for i in sorted(ref_files):
        f = open(i,'r')
        s = ''
        for k in f.readlines():
                s = s + k.strip().replace('\n','')
	#s = s + ' <eos>'
        print(s,file=ref)



dec.close()
ref.close()
