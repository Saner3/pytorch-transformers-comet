import pickle, sys
from pprint import pprint
pkl_file = open(str(sys.argv[1]), 'rb')

data1 = pickle.load(pkl_file)
print(data1)

pkl_file.close()
