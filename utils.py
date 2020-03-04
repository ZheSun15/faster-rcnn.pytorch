# --------------------------------------------------
# Load dicom images from dataset.
# Will delete zipped file after unzip automatically.
# Save dicom images into the type of image_list
#
# zhe @ 20200119
# CT project with LTU
# --------------------------------------------------

from zipfile import ZipFile
import glob
import shutil
import os
import pydicom
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.ndimage
from xml.dom import minidom
from lxml import etree, objectify
from skimage import measure
import pickle
import cv2
from collections import Counter


def auto_unzip(zip_name, input_dir):
    # automatically unzip file if the file we need is zipped.

    # create folder to save files in the zipfile.
    output_dir = '{0}/{1}'.format(input_dir, zip_name.replace('.zip', ''))
    # check if already unzipped
    if os.path.exists(output_dir):
        return 0

    # # copy zipfile to the folder to be extracted to.
    # shutil.copyfile(f'{input_dir}/{zip_name}', f'{output_dir}/{zip_name}')
    # zip_file = f'{output_dir}/{zip_name}'
    zip_file = f'{input_dir}/{zip_name}'

    with ZipFile(zip_file, 'r') as zipf:
        # extract all files in the zip to its current directory
        print('Extracting all the files...')
        for fileM in zipf.namelist():
            zipf.extract(fileM, input_dir)
        print('Unzipping Finished.')

    # delete zip file (both the copied one and the original one)
    # os.remove(zip_file)
    # os.remove(f'{input_dir}/{zip_name}')

    return 0


# from zhihu
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # int16 should be enouth, as all HU value should be less than 32k
    image = image.astype(np.int16)

    # set all -2000 to -1000 (-2000 refers to pixels that are outside the machine's sensor region, which could be consider as 'air')
    image[image <= -2000] = -1000

    # normalize intercept and slope
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def resample_slice(img_cube, slices, new_resolution=[1, 1, 1], data_type='PATIENT_DICOM'):

    # get current resolytion as (z, x, y)
    current_resolution = np.array([slices[0].SliceThickness] + list(slices[0].PixelSpacing), dtype=np.float32)
    img_shape = np.array(img_cube.shape)
    if img_shape.shape[0] == 2:
        current_resolution = current_resolution[1:2]
        new_resolution = new_resolution[1:2]

    resample_seed = current_resolution / new_resolution
    shape_after_resample = np.round(img_shape * resample_seed)
    # print(f'shape_after_resample = {shape_after_resample}')
    # print(f'resolution before resample (z, x, y) = {current_resolution}')
    resize_seed = shape_after_resample / img_shape
    resolution_after_resample = current_resolution / resize_seed

    if 'MASK' in data_type:
        images = scipy.ndimage.interpolation.zoom(img_cube, resize_seed, order=0, mode='nearest')
    else:
        images = scipy.ndimage.interpolation.zoom(img_cube, resize_seed, mode='nearest')
    # print(f'images shape after resample = {images.shape}')

    return np.array(images, dtype=np.int16), resolution_after_resample


def img_normalize(image, min_max=(-1000.0, 400.0)):
    MIN_BOUND = min_max[0]
    MAX_BOUND = min_max[1]

    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.

    return image


def img_normalize_rgb(image, min_max=(-1000.0,400.0), mode=1):
    
    # cut HU with min_max, then map to rgb
    if mode == 1:
        MIN_BOUND = min_max[0]
        MAX_BOUND = min_max[1]
        image[image>MAX_BOUND] = MAX_BOUND - 1
        image[image<MIN_BOUND] = MIN_BOUND

        # now image~[-1000, 400). Divide rgb space to 14x10x10. Calculate constellation diagram.
        rs = np.zeros(14)
        gs = np.zeros(10)
        bs = np.zeros(10)
        constellation = [rs, gs, bs]
        index_nums = [14, 10, 10]
        for i in range(3):
            piece_len = 256 / index_nums[i]
            start = int(piece_len / 2)
            for j in range(index_nums[i]):
                constellation[i][j] = start + j*piece_len

        # map HU to rgb
        rgb_img = np.zeros((image.shape[0], image.shape[1], 3))
        HU = -1000
        for r in range(14):
            for g in range(10):
                for b in range(10):
                    _y, _x = np.where(image==HU)
                    rgb = (constellation[0][r], constellation[1][g], constellation[2][b])
                    rgb_img[_y, _x, :] = np.array(rgb)
                    HU += 1

    # use unmodified HU values, map to rgb
    elif mode == 2:
        image[image<-1000] = -1000

        # now image~[-1000, 1600). Divide rgb space to 12x15x20. Calculate constellation diagram.
        rs = np.zeros(12)
        gs = np.zeros(15)
        bs = np.zeros(20)
        constellation = [rs, gs, bs]
        index_nums = [12, 15, 20]
        for i in range(3):
            piece_len = 256 / index_nums[i]
            start = int(piece_len / 2)
            for j in range(index_nums[i]):
                constellation[i][j] = start + j*piece_len

        # map HU to rgb
        rgb_img = np.zeros((image.shape[0], image.shape[1], 3))
        HU = -1000
        for r in range(12):
            for g in range(15):
                for b in range(20):
                    _y, _x = np.where(image==HU)
                    rgb = (constellation[0][r], constellation[1][g], constellation[2][b])
                    rgb_img[_y, _x, :] = np.array(rgb)
                    HU += 1

    # check element number
    image_ele_num = np.unique(image)
    image_ele_num = image_ele_num.shape[0]
    rgb_cnt = Counter()
    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            px = tuple(np.squeeze(rgb_img[h, w, :]))
            rgb_cnt[px] += 1
    print('origin image elements = %d' % image_ele_num)
    print('rgb image element = {0}'.format(len(rgb_cnt)))
    print('origin image bg = {0}'.format(image[5][5]))
    print('rgb image bg = {0}'.format(rgb_img[5][5]))

    return rgb_img / 255.



def get_dicom_per_folder(folder_pth, save_flag=0, min_max_bound=(-1000, 400), data_type='PATIENT_DICOM'):

    if '.zip' in folder_pth:  # skip zipped files
        return 0

    # print('start with {0}'.format(folder_pth))
    
    # to sort slice names, use a special way to avoid sorted result like: 1, 10, 100, ..., 11, 110, ..., 2, 20, ...
    # no need to do this for son_folder_pths and organ_pths
    ct_names = os.listdir(folder_pth)
    ct_names.sort(key=lambda x:int(x[6:]))
    for i in range(ct_names.__len__()):
        ct_names[i] = os.path.join(folder_pth, ct_names[i])
    # print('ct names after os.listdir and .sort() = ')
    # print(ct_names)
    slices = [pydicom.dcmread(ct_name) for ct_name in ct_names]
    # print('ct lenth = {0}'.format(slices.__len__()))
    # print('the first slice:')
    # print(slices[0])

    # extract hu value
    patient_imgs = get_pixels_hu(slices)
    # print(f'slices before resample = {patient_imgs.shape[0]}')
    # print('min = {0}'.format(np.min(patient_imgs[:,:,:])))
    # print('max = {0}'.format(np.max(patient_imgs[:,:,:])))
    # print('elements = {0}'.format(np.unique(patient_imgs[:,:,:])))
    # for i in range(patient_imgs.shape[0]):
    #     plt.imshow(np.squeeze(patient_imgs[i,:,:]), cmap=plt.cm.bone)
    #     plt.savefig('/home/lydia/code/Zhe/CT/code/ct_visualize/before_res_{0}.png'.format(ct_names[i].replace(folder_pth + '/', '')))

    # resample
    resampled_imgs, resolution_after_resample = resample_slice(patient_imgs, slices, [1, 1, 1], data_type=data_type)
    # print('max = {0}, min = {1}'.format(np.max(resampled_imgs[i]), np.min(resampled_imgs[i])))
    # print(f'slices after resample = {resampled_imgs.shape[0]}')
    # print(f'resolution after resample = {resolution_after_resample}')
    # print('elements after resample = {0}'.format(np.unique(resampled_imgs[:,:,:])))
    # for i in range(patient_imgs.shape[0]):
    #     plt.imshow(np.squeeze(resampled_imgs[i,:,:]), cmap=plt.cm.bone)
    #     plt.savefig('/home/lydia/code/Zhe/CT/code/ct_visualize/after_res_{0}.png'.format(i))  #.format(ct_names[i].replace(folder_pth + '/', '')))
    #     print('ct name = ' + ct_names[i].replace(folder_pth + '/', ''))

    # continue_flag = input('continue? y/n')

    # save as png
    if save_flag:
        for i in range(resampled_imgs.shape[0]):

            # visualize as png
            img_name = 'image_{0}'.format(str(i).zfill(3))
            root_name, folder_name = folder_pth.rsplit('/', 1)
            resample_folder_name = 'RESAMPLE_' + folder_name.replace('DICOM', 'JPEG')
            if not os.path.exists('{0}/{1}'.format(root_name, resample_folder_name)):
                os.mkdir('{0}/{1}'.format(root_name, resample_folder_name))
            save_pth = '{0}/{1}/{2}.jpg'.format(root_name, resample_folder_name, img_name)
            
            # # save to VOC JPEGImages
            # _, patient_name, _ = folder_pth.rsplit('/', 2)
            # img_name = '{0}_{1}'.format(patient_name, str(i).zfill(6))
            # save_pth = f'{jpegimgs_pth}/{img_name}.jpg'
            
            # nomalize image to [0, 1]
            norm_img = img_normalize(np.squeeze(resampled_imgs[i, :, :]), min_max_bound)
            # norm_img = img_normalize_rgb(np.squeeze(resampled_imgs[i, :, :]), min_max_bound, mode=1)
            # print('norm img value number = {0}'.format(np.unique(norm_img)))
            print(norm_img.shape.__len__())
            save_img = Image.fromarray(np.uint8(norm_img * 255.))
            # convert to grayscal or RGB format
            #if norm_img.shape.__len__() == 2:
            #    save_img.convert("L")
            #elif norm_img.shape.__len__() == 3:
            #    save_img.convert("RGB")
            save_img.save(save_pth)
            print(f'img saved on {save_pth}')

        print('all images saved.')
        continue_flag = input('continue? y/n')

    return resampled_imgs


def search_organ_name(label):

    for key in label_dict.keys():
        if label_dict[key] == label:
            return key

    # print('Error! Can\'t find this organ = %d' % label)
    return ''


def getbbox(label, mask_map, filename=None, organ_name=None):

    bbox = []
    h_w = mask_map.shape
    
    '''
    # consider all region as one object
    # bbox should be arranged as (xmin, ymin, xmax, ymax)
    _bbox = np.zeros(4)
    idx = np.where(mask_map == label)

    _bbox[0] = np.min(idx[1])  # xmin
    _bbox[1] = np.min(idx[0])  # ymin
    _bbox[2] = np.max(idx[1])  # xmax
    _bbox[3] = np.max(idx[0])  # ymax
    bbox.append(_bbox.astype(np.int32))
    '''

    # consider different region as different objects
    # seperate disconnected area
    cur_label_mask = mask_map.copy()
    cur_label_mask[cur_label_mask != label] = 0  # set other region as background
    labeled_reg, reg_num = measure.label(cur_label_mask, connectivity=1, return_num=True)
    # print('elements of labeled_reg: {0}'.format(np.unique(labeled_reg)))
    for i in range(1, reg_num + 1):
        _bbox = np.zeros(4)
        _idx = np.where(labeled_reg == i)
        _bbox[0] = np.min(_idx[1])  # xmin
        _bbox[1] = np.min(_idx[0])  # ymin
        _bbox[2] = np.max(_idx[1])  # xmax
        _bbox[3] = np.max(_idx[0])  # ymax
        # check that there's no negative index, nor max < min
        for i in range(_bbox.shape[0]):
            if _bbox[i] < 0:
                _bbox[i] = 0
                print('bbox %d is negative.' % i)
            elif _bbox[i] > h_w[(i+1)%2]:
                _bbox[i] = h_w[(i+1)%2]
                print('bbox %d exceed image size.' % i)
        for i in range(2):
            if _bbox[i+2] < _bbox[i]:
                _bbox[i] = _bbox[i+2] - 1
                print('bbox %d < %d.' % (i+2, i))

        bbox.append(_bbox.astype(np.int32))

    # # visualize bbox    
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.imshow(labeled_reg * 255, cmap=plt.cm.bone)
    # h = (_bbox[3]-_bbox[1]).astype(int)
    # w = (_bbox[2]-_bbox[0]).astype(int)
    # rect = plt.Rectangle((_bbox[0].astype(int),_bbox[1].astype(int)), w, h, fill=False, edgecolor='red', linewidth=2)
    # ax.add_patch(rect)
    # plt.savefig('/home/lydia/code/Zhe/CT/code/ct_visualize/bbox.png')
    # plt.close()

    return bbox


def cube2voc(mask_map, filename, dataset_name, save_flag=0):

    patient_name, img_name = filename.rsplit('_', 1)
    slice_num, _ = img_name.rsplit('.', 1)

    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder(dataset_name),
        E.filename(filename),
        E.source(
            E.database('{0} Database'.format(dataset_name)),
            E.annotation('PASCAL VOC2007 like'),
            E.image(patient_name),
            E.flickrid(slice_num)
        ),
        E.size(
            E.width(mask_map.shape[0]),
            E.height(mask_map.shape[0]),
            E.depth(1)
        ),
        E.segmented(0)
    )

    labels = np.unique(mask_map)
    # print('label of this map:')
    # print(labels)

    im = cv2.imread(f'{jpegimgs_pth}/{filename}')

    for i in range(labels.__len__()):
        organ_name = search_organ_name(labels[i])
        if organ_name == '':
            continue
        
        # print('get bbox for label = %d' % labels[i])
        bbox = getbbox(labels[i], mask_map, filename=filename, organ_name=organ_name)
        # print('elements of mask_map after getting bbox = {0}'.format(np.unique(mask_map)))

        if organ_name == 'lungs':
            print('lungs bbox number = {0}'.format(bbox.__len__()))

        for i in range(bbox.__len__()):
            _bbox = bbox[i]
            _organ_name = organ_name
            
            if organ_name == 'lungs':
                print('i = {0}'.format(i))
                if _bbox[0] < (mask_map.shape[1]/2):  # bbox lies on the left side of the image, consider as rightlung (check patient 1.7 for details)
                    _organ_name = 'rightlung'
                    print('xmin = {0}'.format(_bbox[0]))
                    print('xmax = {0}'.format(_bbox[2]))
                    print('assigned as rightlung.')
                else:
                    _organ_name = 'leftlung'
                    print('xmin = {0}'.format(_bbox[0]))
                    print('xmax = {0}'.format(_bbox[2]))
                    print('assigned as leftlung.')

            _E = objectify.ElementMaker(annotate=False)
            _anno_tree = _E.object(
                _E.name(_organ_name),
                _E.difficult(0),
                _E.bndbox(
                    _E.xmin(_bbox[0]),
                    _E.ymin(_bbox[1]),
                    _E.xmax(_bbox[2]),
                    _E.ymax(_bbox[3])
                )
            )
            anno_tree.append(_anno_tree)
            # print('add %d bbox' % i)
            _bbox = tuple(_bbox)
            cv2.rectangle(im, _bbox[0:2], _bbox[2:4], (0, 204, 0), 2)
            cv2.putText(im, '%s' % (organ_name), (_bbox[0], _bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)

    if save_flag:
        # save xml
        save_path = '{0}/{1}'.format(anno_pth, filename.replace('.jpg', '.xml'))
        etree.ElementTree(anno_tree).write(save_path, pretty_print=True)
        # visualize bbox
        save_bbox_pth = f'{dir_path}/{patient_name[0:9]}/{patient_name}/BBOX/{img_name}'
        if not os.path.exists(f'{dir_path}/{patient_name[0:9]}/{patient_name}/BBOX'):
            os.mkdir(f'{dir_path}/{patient_name[0:9]}/{patient_name}/BBOX')
        cv2.imwrite(save_bbox_pth, im)
        # print('annotation saved to {0}'.format(filename.replace('.jpg', '.xml')))

    return anno_tree


def process_dicom(dir_path, data_type, min_max_bound, dataset_name='3Dircadb'):

    # clear trainval.txt
    if ('MASK' in data_type) and save_flag:
        f = open(trainval_pth, 'w')
        f.close()
        print('Clear all for trainval.txt \n Start processing mask dicoms.')

    root_folder_pths = glob.glob(f'{dir_path}/3Dircadb*')  # root folder name like /3Dircadb1
    root_folder_pths.sort()

    # skip patient 1.3
    patient_to_ignore = [
        "3Dircadb1.3"
    ]

    for root_folder_pth in root_folder_pths:
        
        if '.zip' in root_folder_pth:
            zip_name = root_folder_pth.split('/')[-1]
            auto_unzip(zip_name=zip_name, input_dir=root_folder_pth.replace(f'/{zip_name}', ''))
            root_folder_pth = root_folder_pth.replace('.zip', '')

        folder_pths = glob.glob(f'{root_folder_pth}/*')
        folder_pths.sort()
        for folder_pth in folder_pths:
            # check if this patient should be ignored
            _, patient_id = folder_pth.rsplit('/', 1)
            if patient_id in patient_to_ignore:
                print(f'ignore patient {patient_id}')
                continue
            print('starting with patient {0}'.format(patient_id))

            son_folder_pths = glob.glob(f'{folder_pth}/*')
            son_folder_pths.sort()

            # check if zipped
            unzip_flag = 0
            for son_folder_pth in son_folder_pths:
                if '.zip' in son_folder_pth:
                    # print('Unzipping file ' + son_folder_pth.replace(folder_pth + '/', ''))
                    auto_unzip(son_folder_pth.replace(folder_pth + '/', ''), folder_pth)
                    unzip_flag = 1
            # if unzip has been done, reload file names as unzip operation might create new folder and delete some file
            if unzip_flag:
                son_folder_pths = glob.glob(f'{folder_pth}/*')
                son_folder_pths.sort()

            for son_folder_pth in son_folder_pths:
                if data_type in son_folder_pth:
                    
                    # skip .zip files
                    if '.zip' in son_folder_pth:
                        continue

                    if 'PATIENT' in data_type:
                        _ = get_dicom_per_folder(folder_pth=son_folder_pth, save_flag=save_flag, min_max_bound=min_max_bound)
                    else:
                        organ_folder_pths = glob.glob(son_folder_pth + '/*')
                        organ_folder_pths.sort()
                        mask_cubes = []

                        for organ_folder_pth in organ_folder_pths:
                            
                            # convert the 255s in mask to the corresponding label value(check from google slide)
                            _, organ = organ_folder_pth.rsplit('/', 1)
                            if organ in organ_to_ignore:
                                print(f'skip organ: {organ}.')
                                continue
                            organ_label = label_dict[organ]

                            if organ == 'lungs':
                                print(organ)
                                continue_flag = input('continue? y/n')
                            
                            # get masks of all organs respectively
                            mask_cube = get_dicom_per_folder(folder_pth=organ_folder_pth, save_flag=0, min_max_bound=min_max_bound, data_type=data_type)
                            mask_cube[mask_cube == 255] = organ_label
                            mask_cubes.append(mask_cube)
                            # print('mask elements = {0}'.format(np.unique(mask_cube)))

                        whole_cubes = np.array(mask_cubes)
                        # print('whole cubes size = {0}'.format(whole_cubes.shape))
                        # print('elements = {0}'.format(np.unique(whole_cubes)))
                        labeled_cube = np.max(whole_cubes, axis=0)
                        # print('labeled cubes size (z, x, y) = {0}'.format(labeled_cube.shape))
                        # print('elements of labeled cubes = {0}'.format(np.unique(labeled_cube)))
                        
                        # # visualize as png
                        # for i in range(labeled_cube.shape[0]):
                        #     if not os.path.exists(f'{folder_pth}/MASKS_PNG'):
                        #         os.mkdir(f'{folder_pth}/MASKS_PNG')
                        #     save_pth = '{0}/MASKS_PNG/{1}.png'.format(folder_pth, str(i).zfill(3))
                        #     save_im = Image.fromarray(np.uint8(np.squeeze(labeled_cubes[i,:,:])))
                        #     # convert to RGB format
                        #     save_im.convert("RGB")
                        #     save_im.save(save_pth)
                        #     print('image saved.')

                        for i in range(labeled_cube.shape[0]):
                            filename = '{0}_{1}.jpg'.format(patient_id, str(i).zfill(6))
                            voc_xml = cube2voc(np.squeeze(labeled_cube[i, :, :]), filename=filename, dataset_name=dataset_name, save_flag=save_flag)

                            # save annotation file name into trainval.txt
                            with open(trainval_pth, 'a') as f:
                                f.write(filename.replace('.jpg', '') + '\n')

                        # continue_flag = input('continue? y/n')

    return 0

#----------------------------------------
# configurations

# load label dictionary
pickle_pth = '/home/lydia/code/Zhe/CT/code/cls_name.pkl'
with open(pickle_pth, 'rb') as f:
    label_dict = pickle.load(f)

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
    "rightsurrenalGland",
    "uterus",
    "venacava",
    "leftsurretumor",
    "rightsurretumor"
]

dir_path = '/home/lydia/code/Zhe/CT/toy_dataset' # dir on server 10.176.46.27
data_type = 'MASKS_DICOM'
# dataset_name = '3Dircadb'
dataset_name = 'VOC2007'
dataset_path = '/home/lydia/code/Zhe/CT/code/faster-rcnn.pytorch/data/VOCdevkit2007/VOC2007'
jpegimgs_pth = f'{dataset_path}/JPEGImages'
anno_pth = f'{dataset_path}/Annotations'
trainval_pth = f'{dataset_path}/ImageSets/Main/trainval.txt'
min_max_bound = (-1000.0, 400.0)
save_flag = 0
#--------------------------------------------------


def main():

    # check if dataset path exists.
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    if not os.path.exists(jpegimgs_pth):
        os.mkdir(jpegimgs_pth)

    if not os.path.exists(anno_pth):
        os.mkdir(anno_pth)

    trainval_main, _ = trainval_pth.rsplit('/', 1)
    trainval_imgsets, _ = trainval_main.rsplit('/', 1)
    if not os.path.exists(trainval_main):
        if not os.path.exists(trainval_imgsets):
            os.mkdir(trainval_imgsets)
        os.mkdir(trainval_main)

    # preprocess dicom data
    process_dicom(dir_path, data_type, min_max_bound, dataset_name)


if __name__ == '__main__':
    main()

