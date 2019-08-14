import pickle, sys, argparse
from pprint import pprint
sys.path.append(".")
from ckbc_demo.demo_bilinear import Scorer

parser = argparse.ArgumentParser()
parser.add_argument("--print_negative", action='store_true', help="print those triples that are considered negative by the scorer.")
parser.add_argument("--cmp_novelty", action='store_true', help="compare the novelty.")
parser.add_argument("--input_file", type=str, required=True, help="generation results file")
args = parser.parse_args()

pkl_file = open(args.input_file, 'rb')
train_set = open("data/conceptnet/train100k.txt", "r")
train_set = train_set.read().splitlines()
train_triples = set()
train_objects = set()
train_sr = set()
for pair in train_set:
    line = pair.split('\t')
    r = "".join(line[0].lower().split())
    e1 = line[1]
    e2 = line[2]
    train_triples.add((e1, r, e2))
    train_objects.add(e2)
    train_sr.add((e1, r))

data = pickle.load(pkl_file)
scorer = Scorer()
positive_prediction = 0
negative_prediction = 0
novelsro = 0
novelo = 0
novelsr = 0
print("The Scorer does not support those relations:\n", " ".join( ['HasPainCharacter', 'HasPainIntensity', 'LocationOfAction', 'LocatedNear',
'DesireOf','NotMadeOf','InheritsFrom','InstanceOf','RelatedTo','NotDesires',
'NotHasA','NotIsA','NotHasProperty','NotCapableOf']))
print("\nPairs that are considered incorrect:\n")
for pair in data:
    try:
        results = scorer.gen_score(pair['e1'], pair['sequence'])
    except:
        print("error:", pair)
        exit()
    rel = "".join(pair['r'].split())    # lower, no space,
    # COMET code mis-spells "hasprerequisite"
    if rel == "hasprequisite":
        rel = "hasprerequisite"
    thispair = (pair['e1'], rel, pair['sequence'])
    if thispair not in train_triples:
        novelsro += 1
        # PRINT OUT TO CHECK IF IT IS REALLY NOVEL
        if args.cmp_novelty:
            print("\nGENERATED:", pair['e1'], pair['r'], pair['sequence'])
            for triple in train_triples:
                _rel = "".join(pair['r'].split())
                if _rel == "hasprequisite":
                    _rel = "hasprerequisite"
                if triple[0] == pair['e1'] and triple[1] == _rel:
                    print("   INFILE:", pair['e1'], pair['r'], triple[2])
    if pair['sequence'] not in train_objects:
        novelo += 1
    if (pair['e1'], rel) not in train_sr:
        novelsr += 1
    
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

valid_data_num = positive_prediction + negative_prediction
total_data_num = len(data)
print("score", positive_prediction / valid_data_num)
print("N/Tsro", novelsro / total_data_num)
print("N/To", novelo / total_data_num)
print("N/Tsr", novelsr / total_data_num)
pkl_file.close()

