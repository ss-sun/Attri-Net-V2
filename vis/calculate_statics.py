



result = {
    "bbox": {
        "Atelectasis": "0.9139155698512671",
        "Cardiomegaly": "0.9506671358990734",
        "Consolidation": "0.9334858408012374",
        "Edema": "0.946014588666719",
        "Effusion": "0.9657238972659846"
    },
    "pseudo_mask": {
        "Atelectasis": "0.9147297484319268",
        "Cardiomegaly": "0.9514335485067497",
        "Consolidation": "0.9434728298146928",
        "Edema": "0.9491108293277773",
        "Effusion": "0.9601062513533383"
    },
    "pseudo_bbox": {
        "Atelectasis": "0.9318604910494599",
        "Cardiomegaly": "0.9403858318478595",
        "Consolidation": "0.9538145008645678",
        "Edema": "0.9538192034700974",
        "Effusion": "0.9719424492535257"
    },
    "mixed": {
        "Atelectasis": "0.9538388153962564",
        "Cardiomegaly": "0.9710226569382138",
        "Consolidation": "0.9584078673155213",
        "Edema": "0.9719850630934123",
        "Effusion": "0.9682384358120661"
    }
}

for key in result.keys():
    print(key)
    sum=0
    num=0
    for key2 in result[key].keys():
        print(key2, result[key][key2])
        if float(result[key][key2]) != 0:
            sum += float(result[key][key2])
            num += 1
    print("avg: ")
    print(sum/num)
    print("\n\n")