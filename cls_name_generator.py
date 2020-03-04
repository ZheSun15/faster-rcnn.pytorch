# ------------------
# generate class name
# copy the output, paste on the according lines in 3Dircadb.py where defines the class name
# zhe @ 2020.02.11
# ------------------

import numpy as np
import pandas as pd
import pickle


def search_organ_name(label):

    organs = []
    return_flag = 0
    
    for key in label_dict.keys():
        if label_dict[key] == label:
            return_flag = 1
            organs.append(key)

    if return_flag:
        return organs
    else:
        print('Error! Can\'t find this organ = %d' % label)
        return []


def gen_cls_name(save_pth):

    label_values = []
    cls_names = []
    names2txt = []
    
    # load all values
    for key in label_dict.keys():
        if key in organ_to_ignore:
           continue
        label_values.append(label_dict[key])
    
    label_values = sorted(label_values)
    label_values = np.unique(label_values)

    # sort class names according to label value
    new_cls_dict = {}
    new_label_value = 1
    for label_value in label_values:
        cls_name = search_organ_name(label_value)
        cls_names.extend(cls_name)
        names2txt.append('\'{0}\','.format(cls_name[0]))  # combine organs that have same label value.
        # create dict for util.py to search value by organ name
        for t_name in cls_name:
            new_cls_dict[t_name] = new_label_value
        new_label_value += 1

    # save to txt
    save_txt_pth = save_pth + '.txt'
    with open(save_txt_pth, 'w') as f:
        for i in range(names2txt.__len__()):
            f.write(names2txt[i])
    # save to excel
    save_excel_pth = save_pth + '.xlsx'
    excel_data = []
    for organ in new_cls_dict.keys():
        t_dict = {
            'cls_name': organ,
            'origin_label_value': label_dict[organ],
            'label_value': new_cls_dict[organ]
        }
        excel_data.append(t_dict)
    df = pd.DataFrame(data=excel_data, columns=['cls_name', 'origin_label_value', 'label_value'])
    df.to_excel(save_excel_pth)
    # save to pickle
    save_pickle_pth = save_pth + '.pkl'
    print(new_cls_dict)
    with open(save_pickle_pth, 'wb') as f:
        pickle.dump(new_cls_dict, f, pickle.HIGHEST_PROTOCOL)

    return 0



# label dictionary
label_dict = {
    "aorta": 74,
    "artery": 75,
    "biliarysystem": 3,
    "bladder": 4,
    "bone": 5,
    "colon": 6,
    "duodenum": 7,
    "gallbladder": 8,
    "heart": 9,
    "Hyperplasie": 10,
    "kidneys": 11,
    "leftkidney": 12,
    "rightkidney": 13,
    "liver": 20,
    "livercyst": 21,
    "liverkyste": 21,
    "liverkyst": 21,
    "leftlung": 30,
    "lungs": 30,
    "rightlung": 31,
    "lymphNodes": 40,
    "metal": 41,
    "metastasectomie": 42,
    "pancreas": 43,
    "portalvein": 73,
    "portalvein1": 73,
    "sigmoid": 46,
    "skin": 47,
    "smallintestin": 48,
    "spleen": 49,
    "stomach": 50,
    "stomachoesophage": 51,
    "Stones": 52,
    "surrenalgland": 60,
    "leftsurrenalgland": 61,
    "rightsurrenalgland": 62,
    "uterus": 70,
    "venacava": 71,
    "venoussystem": 72,
    "tumor": 100,
    "livertumor": 100,
    "livertumors": 100,
    "livertumor01": 100,
    "livertumor02": 100,
    "livertumor03": 100,
    "livertumor04": 100,
    "livertumor05": 100,
    "livertumor06": 100,
    "livertumor07": 100,
    "livertumor1": 100,
    "livertumor2": 100,
    "leftsurretumor": 101,
    "rightsurretumor": 102
}

# only consider common organs, ignore the rest
organ_to_ignore = [
    "biliarysystem",
    "bladder",
    "bone",
    "colon",
    "duodenum",
    "Hyperplasie",
    "kidneys",
    "livercyst",
    "liverkyste",
    "liverkyst",
    "lymphNodes",
    "metal",
    "metastasectomie",
    "sigmoid",
    "skin",
    "smallintestin",
    "stomach",
    "stomachoesophage",
    "Stones",
    "surrenalgland",
    "leftsurrenalgland",
    "rightsurrenalgland",
    "uterus",
    "venacava",
    "leftsurretumor",
    "rightsurretumor"
]



if __name__ == '__main__':
    
    save_pth = '/home/lydia/code/Zhe/CT/code/cls_name'
    gen_cls_name(save_pth)
