# load detection result (x.pkl file) and save as .txt file
# zhe @ 2020.03
# with LTU

import pickle

def pkl2txt(label_list, img_name_list, pkl_pth, txt_pth, threshold):

    with open(pkl_pth, 'rb') as f:
        detection = pickle.load(f)

    
    txt_f = open(txt_pth, 'w')

    # detection is a list, the number of elements of which equals to total class number
    # each element is a list which contains 943 elements(=test image number)
        
    print('detection len = {0}'.format(detection.__len__()))
    print('cur_detection len = {0}'.format(detection[0].__len__()))

    for i in range(detection.__len__() - 1):
        cur_class_name = label_list[i]
        cur_class_detection = detection[i]
        if cur_class_detection.__len__() == 0:
            continue
        for img_num in range(cur_class_detection.__len__()):
            cur_detection = cur_class_detection[img_num]
            img_name = img_name_list[img_num]
            if cur_detection.__len__() == 0:
            	continue
            for obj_num in range(cur_detection.shape[0]):
	            _confidence = cur_detection[obj_num][4]
	            # filter object with low confidence
	            if _confidence < threshold:
	            	continue
	            bbox = cur_detection[obj_num][:4]
	            _bbox = '{0},{1},{2},{3}'.format(bbox[0],bbox[1],bbox[2],bbox[3])
	            _dete = '{0},{1},{2},{3}\n'.format(img_name,cur_class_name, _confidence, _bbox)
	            txt_f.write(_dete)

    txt_f.close()

    print('txt saved.')



    return 0


def main():

    label_list_pth = '/home/lydia/code/Zhe/CT/code/cls_name.txt'
    img_list_pth = '/storage/lydia/3Dircadb/ImageSets/Main/test.txt'
    threshold = 0.3
    pkl_pth = '/home/lydia/code/Zhe/CT/code/faster-rcnn.pytorch/output/vgg16/voc_2007_test/faster_rcnn_10/detections.pkl'
    txt_pth = '/home/lydia/code/Zhe/CT/code/faster-rcnn.pytorch/output/vgg16/voc_2007_test/faster_rcnn_10/detections.txt'
    
    label_list = []
    with open(label_list_pth, 'r') as f:
        for line in f:
            label_list.append(str(line.strip('\n')))
        line = label_list[0]
        labels = line.split(',')
        label_list = []
        for label in labels:
        	label_list.append(str(label.strip('\"')))
    
    img_name_list = []
    with open(img_list_pth, 'r') as f:
        for line in f:
            img_name_list.append(str(line.strip('\n')))

    pkl2txt(label_list, img_name_list, pkl_pth, txt_pth, threshold)


if __name__ == '__main__':
    main()
