import json
dic = {}
with open ("FB-data", "r") as f:
    data = f.read().split()
    for line in data:
        dic[line] = ""

dicstr = json.dumps(dic, indent=4)
print(dicstr)