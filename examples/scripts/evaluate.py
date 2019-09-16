import pickle, sys, argparse, random, io
from pprint import pprint
import numpy as np
sys.path.append(".")
from ckbc_demo.demo_bilinear import Scorer
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument("--print_negative", action='store_true', help="print those triples that are considered negative by the scorer.")
parser.add_argument("--calc_similarity", action='store_true', help="compare the novelty.")
parser.add_argument("--rel_match", action='store_true', help="compare if the rel matches")
parser.add_argument("--input_file", type=str, required=True, help="generation results file")
args = parser.parse_args()

pkl_file = open(args.input_file, 'rb')
train_set = open("data/conceptnet/train100k_CN.txt", "r")
test_set = open("data/conceptnet/test_CN.txt", "r")

train_set = train_set.read().splitlines()
test_set = test_set.read().splitlines()

train_triples = set()
train_objects = set()
train_sr = {}
test_sr = {}

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data

def get_avg_vec(fasttext_vecs, phrase):
    phrase = phrase.split()
    try:
        vecs = np.array([fasttext_vecs[word] for word in phrase])
    except:
        return None
    return np.mean(vecs, axis=0)

for pair in train_set:
    line = pair.split('\t')
    r = "".join(line[0].lower().split())
    e1 = line[1]
    e2 = line[2]
    train_triples.add((e1, r, e2))
    train_objects.add(e2)
    try:
        train_sr[(e1, r)].append(e2)
    except:
        train_sr[(e1, r)] = [e2]

similarity_with_train_set = []
similarity_with_test_set = []
if args.calc_similarity:
    print("loading fasttext word vectors ...")
    fasttext_vectors = load_vectors("wiki-news-300d-1M.vec")
    print("loading done ...")
data = pickle.load(pkl_file)
scorer = Scorer()
positive_prediction = 0
negative_prediction = 0
novelsro, novelo, novelsr = 0, 0, 0
print("The Scorer does not support those relations:\n", " ".join( ['HasPainCharacter', 'HasPainIntensity', 'LocationOfAction', 'LocatedNear',
'DesireOf','NotMadeOf','InheritsFrom','InstanceOf','RelatedTo','NotDesires',
'NotHasA','NotIsA','NotHasProperty','NotCapableOf']))
print("\nPairs that are considered incorrect:\n")

match_rel = 0
for pair in data:
    e1 = pair['e1']
    e2 = pair['sequence']
    ref = pair['reference']
    try:
        results = scorer.gen_score(e1, e2)
    except:
        print("error:", pair)
        exit()
    rel = "".join(pair['r'].lower().split())    # lower, no space,
    if args.rel_match and rel == "".join(ref.lower().split()):
        match_rel += 1
    # COMET code mis-spells "hasprerequisite"
    if rel == "hasprequisite":
        rel = "hasprerequisite"

    thispair = (e1, rel, e2)
    if args.calc_similarity:
        if (e1, rel) in test_sr.keys():
            test_sr[(e1, rel)][1].append(e2)
            test_sr[(e1, rel)][0].append(ref)
        else:
            test_sr[(e1, rel)] = [[ref], [e2]]

    if thispair not in train_triples:
        novelsro += 1
        # PRINT OUT TO CHECK IF IT IS REALLY NOVEL
        # if args.cmp_novelty:
        #     print("\nGENERATED:", pair['e1'], pair['r'], pair['sequence'])
        #     for triple in train_triples:
        #         _rel = "".join(pair['r'].split())
        #         if _rel == "hasprequisite":
        #             _rel = "hasprerequisite"
        #         if triple[0] == pair['e1'] and triple[1] == _rel:
        #             print("   INFILE:", pair['e1'], pair['r'], triple[2])
    if e2 not in train_objects:
        novelo += 1
    if (e1, rel) not in train_sr.keys():
        novelsr += 1
    if args.calc_similarity and (e1, rel) in train_sr.keys():
        # calc similarity between e2 and train_sr[(e1, rel)]
        targets = train_sr[(e1, rel)]
        e2_vec = get_avg_vec(fasttext_vectors, e2)
        targets_vecs = [get_avg_vec(fasttext_vectors, target) for target in targets]
        targets_vecs = [x for x in targets_vecs if x is not None]   # remove Nones

        if targets_vecs == [] or e2_vec is None:
            continue
        similarities = cosine_similarity([e2_vec], targets_vecs)
        similarity_with_train_set.extend(similarities[0])


    found = False
    correct = False
    for rel_tuple in results:
        # rel_tuple: (relation, score)
        if rel == rel_tuple[0]:
            found = True
            if rel_tuple[1] > 0.50:
                positive_prediction += 1
                correct = True
            else:
                negative_prediction += 1
                if (args.print_negative):
                    print("NEGATIVE:", pair['e1'], pair['r'], pair['sequence'])
            break

if args.calc_similarity:
    for (e1, rel) in test_sr.keys():
        predicts = test_sr[(e1, rel)][1]
        targets = test_sr[(e1, rel)][0]
        predicts_vecs = [get_avg_vec(fasttext_vectors, predict) for predict in predicts]
        targets_vecs = [get_avg_vec(fasttext_vectors, target) for target in targets]
        predicts_vecs = [x for x in predicts_vecs if x is not None]   # remove Nones
        targets_vecs = [x for x in targets_vecs if x is not None]   # remove Nones
        if targets_vecs == [] or predicts_vecs == []:
            continue

        similarities = cosine_similarity(predicts_vecs, targets_vecs)
        for similarity in similarities:
            similarity_with_test_set.extend(similarity)

valid_data_num = positive_prediction + negative_prediction
total_data_num = len(data)
print("score", positive_prediction / valid_data_num)
print("N/Tsro", novelsro / total_data_num)
print("N/To", novelo / total_data_num)
print("N/Tsr", novelsr / total_data_num)
if args.rel_match:
    print("match rels:", match_rel / total_data_num)
if args.calc_similarity:
    print("similarities with train set:", np.mean(similarity_with_train_set))
    print("similarities with ground truth:", np.mean(similarity_with_test_set))
pkl_file.close()