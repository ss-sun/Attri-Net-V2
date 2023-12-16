




result ={
    "chexpert_shap": {
        "Atelectasis": "0.1146365668537041",
        "Cardiomegaly": "0.23603493795071928",
        "Consolidation": "0.17692218854953862",
        "Edema": "0.4037502380495755",
        "Pleural Effusion": "0.16530490021211244"
    },
    "nih_chestxray_shap": {
        "Atelectasis": "0.09825546044536132",
        "Cardiomegaly": "0.5975433892658468",
        "Consolidation": 0,
        "Edema": 0,
        "Effusion": "0.13859717793047396"
    },
    "vindr_cxr_shap": {
        "Aortic enlargement": "0.1053797521892999",
        "Cardiomegaly": "0.16761951488370055",
        "Pulmonary fibrosis": "0.006479853448775065",
        "Pleural thickening": "0.005157800083571712",
        "Pleural effusion": "0.050009986051089805"
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