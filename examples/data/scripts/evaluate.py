import pickle, sys
from pprint import pprint
<<<<<<< HEAD
sys.path.append('./ckbc_demo')
print(sys.path)
from ckbc_demo.demo_bilinear import Scorer

pkl_file = open(str(sys.argv[1]), 'rb')
train_set = open("data/conceptnet/train100k.txt", "r")
=======
from ckbc_demo.demo_bilinear import Scorer
pkl_file = open(str(sys.argv[1]), 'rb')
train_set = open("/nas/home/jsun/comet-commonsense/data/conceptnet/train100k.txt", "r")
>>>>>>> b7b14d583f606144c32547d8aa6b408eb9a733ae
train_set = train_set.read().splitlines()
train_datas = set()
train_objects = set()
train_sr = set()
for pair in train_set:
    line = pair.split('\t')
    r = "".join(line[0].lower().split())
    e1 = line[1]
    e2 = line[2]
    train_datas.add((e1, r, e2))
    train_objects.add(e2)
    train_sr.add((e1, r))

data1 = pickle.load(pkl_file)
scorer = Scorer()
positive_prediction = 0
novelsro = 0
novelo = 0
novelsr = 0
for pair in data1:
    try:
        results = scorer.gen_score(pair['e1'], pair['sequence'])
    except:
        print(pair)
        exit()
    rel = "".join(pair['r'].split())
    if rel == "hasprequisite":
        rel = "hasprerequisite"
    thispair = (pair['e1'], rel, pair['sequence'])
    if thispair not in train_datas:
        novelsro += 1
    if pair['sequence'] not in train_objects:
        novelo += 1
    if (pair['e1'], rel) not in train_sr:
        novelsr += 1
    found = False
    correct = False
    for reltuple in results:
        if rel == reltuple[0]:
            found = True
            if reltuple[1] > 0.50:
                positive_prediction+=1
                correct = True
            break
    if not correct:
        print(pair['e1'], pair['r'], pair['sequence'])
    if not found:
        print(rel)
        #print(results)
        #print("Not found Error")
        #exit()
print("score", positive_prediction / len(data1))
print("N/Tsro", novelsro / len(data1))
print("N/To", novelo / len(data1))
print("N/Tsr", novelsr / len(data1))
pkl_file.close()

