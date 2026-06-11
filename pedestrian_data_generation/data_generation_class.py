import numpy as np
import cv2
import os
import pickle
import sys
from hrnet.SimpleHRNet import SimpleHRNet

def update_progress(progress):
    """
    Shows the progress
    Args:
        progress: Progress thus far
    """
    barLength = 20  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)

    block = int(round(barLength * progress))
    text = "\r[{}] {:0.2f}% {}".format("#" * block + "-" * (barLength - block), progress * 100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def get_path(file_name='',
             sub_folder='',
             save_folder='models',
             dataset='pie',
             save_root_folder='data/'):
    """
    Generates paths for saving model and config data.
    Args:
        file_name: The actual save file name , e.g. 'model.h5'
        sub_folder: If another folder to be created within the root folder
        save_folder: The name of folder containing the saved files
        dataset: The name of the dataset used
        save_root_folder: The root folder
    Return:
        The full path and the path to save folder
    """
    save_path = os.path.join(save_root_folder, save_folder, sub_folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return os.path.join(save_path, file_name), save_path

class data_generation():
    """
    hierfusion MASK_PCPA
    Class init function

    Args:
        num_hidden_units: Number of recurrent hidden layers
        cell_type: Type of RNN cell
        **kwargs: Description
    """

    def __init__(self,
                 **kwargs):
        """
        Class init function

        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
            **kwargs: Description
        """
        super().__init__(**kwargs)
        # Network parameters

    def create_scene_image(self, img_sequences,
                           ped_ids, boxs, file_path,
                           data_type='train',
                           dataset='pie'):
        print('\n#####################################')
        print('Generating scene image %s' % data_type)
        print('#####################################')

        data_all = []
        file_path = file_path

        path_data = os.path.join(file_path, data_type)
        i = -1
        if not os.path.exists(path_data):
            os.makedirs(path_data)
        for seq, pid, box in zip(img_sequences, ped_ids, boxs):
            data_all = []
            i += 1
            update_progress(i / len(img_sequences))
            ss = str(seq[-1]).replace('\\', '/')
            vid_id = ss.split('/')[-2]
            img_name = ss.split('/')[-1].split('.')[0]
            file = os.path.join(path_data, vid_id + '_' + pid[-1][0] + '_' + img_name)
            if not os.path.exists(file):
                os.makedirs(file)
            j = 0
            for imp, p, b in zip(seq, pid, box):
                if not dataset == 'pie':
                    name_o = imp.split('/')[-1]
                    name_l = '0' + imp.split('/')[-1]
                    imp = imp.replace(name_o, name_l)
                image = cv2.imread(imp, cv2.IMREAD_COLOR)

                file_image_path = os.path.join(file, str(j) + '.png')

                cv2.imwrite(file_image_path, image)
                j = j + 1

        return data_all

    def create_scene_image_pedestrain(self, img_sequences,
                                      ped_ids, boxs, file_path,
                                      data_type='train',
                                      dataset='pie'):
        print('\n#####################################')
        print('Generating scene with pedestrain bounding box image %s' % data_type)
        print('#####################################')

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        path_data = os.path.join(file_path, data_type)
        i = -1
        if not os.path.exists(path_data):
            os.makedirs(path_data)
        for seq, pid, box in zip(img_sequences, ped_ids, boxs):
            data_all = []
            i += 1
            update_progress(i / len(img_sequences))
            ss = str(seq[-1]).replace('\\', '/')
            vid_id = ss.split('/')[-2]
            img_name = ss.split('/')[-1].split('.')[0]
            file = os.path.join(path_data, vid_id + '_' + pid[-1][0] + '_' + img_name)
            if not os.path.exists(file):
                os.makedirs(file)
            j = 0
            for imp, p, b in zip(seq, pid, box):
                if not dataset == 'pie':
                    name_o = imp.split('/')[-1]
                    name_l = '0' + imp.split('/')[-1]
                    imp = imp.replace(name_o, name_l)
                image = cv2.imread(imp, cv2.IMREAD_COLOR)

                h, w, _ = image.shape
                b = [int(x) for x in b]

                cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

                file_image_path = os.path.join(file, str(j) + '.png')

                cv2.imwrite(file_image_path, image)
                j = j + 1

        return data_all

    def create_pedestrain_image(self, img_sequences,
                                ped_ids, boxs, file_path,
                                data_type='train',
                                dataset='pie'):
        print('\n#####################################')
        print('Generating pedestrain image %s' % data_type)
        print('#####################################')

        data_all = []
        file_path = file_path

        path_data = os.path.join(file_path, data_type)
        i = -1
        if not os.path.exists(path_data):
            os.makedirs(path_data)
        for seq, pid, box in zip(img_sequences, ped_ids, boxs):
            data_all = []
            i += 1
            update_progress(i / len(img_sequences))
            ss = str(seq[-1]).replace('\\', '/')
            vid_id = ss.split('/')[-2]
            img_name = ss.split('/')[-1].split('.')[0]
            file = os.path.join(path_data, vid_id + '_' + pid[-1][0] + '_' + img_name)
            if not os.path.exists(file):
                os.makedirs(file)
            j = 0
            for imp, p, b in zip(seq, pid, box):
                if not dataset == 'pie':
                    name_o = imp.split('/')[-1]
                    name_l = '0' + imp.split('/')[-1]
                    imp = imp.replace(name_o, name_l)
                image = cv2.imread(imp, cv2.IMREAD_COLOR)

                b = [int(x) for x in b]
                image = image[b[1]:b[3], b[0]:b[2]]
                file_image_path = os.path.join(file, str(j) + '.png')

                cv2.imwrite(file_image_path, image)
                j = j + 1

        return data_all

    def creat_pose(self, img_sequences,
                   ped_ids, boxs, file_path,
                   data_type='train',
                   dataset='pie'):
        print('\n#####################################')
        print('Generating pedestrain pose %s' % data_type)
        print('#####################################')
        data_all = []
        path_data = os.path.join(file_path, data_type)
        i = -1
        if not os.path.exists(path_data):
            os.makedirs(path_data)
        hrnet = SimpleHRNet(48, 17, "./hrnet/weights/pose_hrnet_w48_384x288.pth")

        for seq, pid, box in zip(img_sequences, ped_ids, boxs):
            data_all = []
            i += 1
            update_progress(i / len(img_sequences))
            ss = str(seq[-1]).replace('\\', '/')
            vid_id = ss.split('/')[-2]
            img_name = ss.split('/')[-1].split('.')[0]
            file = os.path.join(path_data, vid_id + '_' + pid[-1][0] + '_' + img_name + '.pkl')
            # file = os.path.join(path_data, vid_id + '_' + pid[-1][0] + '_' + img_name)
            j = 0
            for imp, p, b in zip(seq, pid, box):
                datas = []
                if not dataset == 'pie':
                    name_o = imp.split('/')[-1]
                    name_l = '0' + imp.split('/')[-1]
                    imp = imp.replace(name_o, name_l)
                image = cv2.imread(imp, cv2.IMREAD_COLOR)
                h, w, _ = image.shape
                b = [int(x) for x in b]
                image = image[b[1]:b[3], b[0]:b[2]]
                h_box, w_box, _ = image.shape
                joints = hrnet.predict(image)
                joints = np.array(joints)
                joints = np.squeeze(joints)

                for item in joints:
                    data = []
                    y = item[0]
                    x = item[1]
                    # cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), 1)
                    # cv2.putText(image, str(count), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 255),
                    #             1)
                    s = item[2]
                    x = round(x / w_box, 4)
                    y = round(y / h_box, 4)
                    data.append(x)
                    data.append(y)
                    data.append(s)
                    datas.append(data)
                x_17 = (datas[5][0] + datas[6][0]) / 2
                y_17 = (datas[5][1] + datas[6][1]) / 2
                s_17 = (datas[5][2] + datas[6][2]) / 2

                data = []
                data.append(x_17)
                data.append(y_17)
                data.append(s_17)
                datas.append(data)
                # cv2.circle(image, (int(x_17), int(y_17)), 1, (0, 255, 0), 1)
                data_all.append(datas)
                # file_image_path=os.path.join(file,str(j)+'.png')

                # cv2.imwrite(file_image_path, image)
                j = j + 1
            f_in = open(file, 'wb')
            pickle.dump(data_all, f_in)
            f_in.close()
        return data_all

    def creat_box(self, img_sequences,
                  ped_ids, boxs, file_path,
                  data_type='train',
                  dataset='pie'):
        print('\n#####################################')
        print('Creating bbox %s' % data_type)
        print('#####################################')
        data_all = []
        path_data = os.path.join(file_path, data_type)
        i = -1
        if not os.path.exists(path_data):
            os.makedirs(path_data)
        for seq, pid, box in zip(img_sequences, ped_ids, boxs):
            i += 1
            update_progress(i / len(img_sequences))
            ss = str(seq[-1]).replace('\\', '/')
            vid_id = ss.split('/')[-2]
            img_name = ss.split('/')[-1].split('.')[0]
            file = os.path.join(path_data, vid_id + '_' + pid[-1][0] + '_' + img_name + '.pkl')
            if os.path.exists(file):
                continue
            bboxes = []
            for imp, p, b in zip(seq, pid, box):
                data = []
                if not dataset == 'pie':
                    name_o = imp.split('/')[-1]
                    name_l = '0' + imp.split('/')[-1]
                    imp = imp.replace(name_o, name_l)
                image = cv2.imread(imp, cv2.IMREAD_COLOR)
                h, w, _ = image.shape
                b0 = b[0] / w
                b1 = b[1] / h
                b2 = b[2] / w
                b3 = b[3] / h
                data.append(b0)
                data.append(b1)
                data.append(b2)
                data.append(b3)
                bboxes.append(data)

            f_in = open(file, 'wb')
            pickle.dump(bboxes, f_in)
            f_in.close()
        return data_all

    def creat_label(self, img_sequences,
                    ped_ids, boxs, labels, file_path,
                    data_type='train',
                    dataset='pie'):
        print('\n#####################################')
        print('Creating label %s' % data_type)
        print('#####################################')
        data_all = []
        path_data = os.path.join(file_path, data_type)
        i = -1
        if not os.path.exists(path_data):
            os.makedirs(path_data)
        for seq, pid, box, label in zip(img_sequences, ped_ids, boxs, labels):
            i += 1
            update_progress(i / len(img_sequences))
            ss = str(seq[-1]).replace('\\', '/')
            vid_id = ss.split('/')[-2]
            img_name = ss.split('/')[-1].split('.')[0]
            file_out = os.path.join(path_data, vid_id + '_' + pid[-1][0] + '_' + img_name + '.pkl')
            if os.path.exists(file_out):
                continue
            f_in = open(file_out, 'wb')
            pickle.dump(label, f_in)
            f_in.close()
        return data_all

    def creat_speed(self, img_sequences,
                    ped_ids, boxs, speeds, file_path,
                    data_type='train',
                    dataset='pie'):
        print('\n#####################################')
        print('Creating speed %s' % data_type)
        print('#####################################')
        data_all = []
        path_data = os.path.join(file_path, data_type)
        i = -1
        if not os.path.exists(path_data):
            os.makedirs(path_data)
        for seq, pid, box, speed in zip(img_sequences, ped_ids, boxs, speeds):
            i += 1
            update_progress(i / len(img_sequences))
            ss = str(seq[-1]).replace('\\', '/')
            vid_id = ss.split('/')[-2]
            img_name = ss.split('/')[-1].split('.')[0]
            file_out = os.path.join(path_data, vid_id + '_' + pid[-1][0] + '_' + img_name + '.pkl')
            if os.path.exists(file_out):
                continue
            f_in = open(file_out, 'wb')
            pickle.dump(speed, f_in)
            f_in.close()
        return data_all

    def balance_data_samples(self, d, img_width, balance_tag='crossing'):
        """
        Balances the ratio of positive and negative data samples. The less represented
        data type is augmented by flipping the sequences
        Args:
            d: Sequence of data samples
            img_width: Width of the images
            balance_tag: The tag to balance the data based on
        """
        print("Balancing with respect to {} tag".format(balance_tag))
        gt_labels = [gt[0] for gt in d[balance_tag]]
        num_pos_samples = np.count_nonzero(np.array(gt_labels))
        num_neg_samples = len(gt_labels) - num_pos_samples

        # finds the indices of the samples with larger quantity
        if num_neg_samples == num_pos_samples:
            print('Positive and negative samples are already balanced')
        else:
            print('Unbalanced: \t Positive: {} \t Negative: {}'.format(num_pos_samples, num_neg_samples))
            if num_neg_samples > num_pos_samples:
                gt_augment = 1
            else:
                gt_augment = 0

            num_samples = len(d[balance_tag])
            for i in range(num_samples):
                if d[balance_tag][i][0][0] == gt_augment:
                    for k in d:
                        if k == 'center':
                            flipped = d[k][i].copy()
                            flipped = [[img_width - c[0], c[1]]
                                       for c in flipped]
                            d[k].append(flipped)
                        if k == 'box':
                            flipped = d[k][i].copy()
                            flipped = [np.array([img_width - b[2], b[1], img_width - b[0], b[3]])
                                       for b in flipped]
                            d[k].append(flipped)
                        if k == 'image':
                            flipped = d[k][i].copy()
                            flipped = [im.replace('.png', '_flip.png') for im in flipped]
                            d[k].append(flipped)
                        if k in ['speed', 'ped_id', 'crossing', 'walking', 'looking']:
                            d[k].append(d[k][i].copy())

            gt_labels = [gt[0] for gt in d[balance_tag]]
            num_pos_samples = np.count_nonzero(np.array(gt_labels))
            num_neg_samples = len(gt_labels) - num_pos_samples
            if num_neg_samples > num_pos_samples:
                rm_index = np.where(np.array(gt_labels) == 0)[0]
            else:
                rm_index = np.where(np.array(gt_labels) == 1)[0]

            # Calculate the difference of sample counts
            dif_samples = abs(num_neg_samples - num_pos_samples)
            # shuffle the indices
            np.random.seed(42)
            np.random.shuffle(rm_index)
            # reduce the number of indices to the difference
            rm_index = rm_index[0:dif_samples]

            # update the data
            for k in d:
                seq_data_k = d[k]
                d[k] = [seq_data_k[i] for i in range(0, len(seq_data_k)) if i not in rm_index]

            new_gt_labels = [gt[0] for gt in d[balance_tag]]
            num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
            print('Balanced:\t Positive: %d  \t Negative: %d\n'
                  % (num_pos_samples, len(d[balance_tag]) - num_pos_samples))

    def get_data_sequence(self, data_type, data_raw, opts):
        """
        Generates raw sequences from a given dataset
        Args:
            data_type: Split type of data, whether it is train, test or val
            data_raw: Raw tracks from the dataset
            opts:  Options for generating data samples
        Returns:
            A list of data samples extracted from raw data
            Positive and negative data counts
        """
        print('\n#####################################')
        print('Generating raw data')
        print('#####################################')
        d = {'center': data_raw['center'].copy(),
             'box': data_raw['bbox'].copy(),
             'ped_id': data_raw['pid'].copy(),
             'crossing': data_raw['activities'].copy(),
             'image': data_raw['image'].copy()}

        balance = opts['balance_data'] if data_type == 'train' else False
        obs_length = opts['obs_length']
        time_to_event = opts['time_to_event']
        normalize = opts['normalize_boxes']

        try:
            d['speed'] = data_raw['obd_speed'].copy()
        except KeyError:
            d['speed'] = data_raw['vehicle_act'].copy()
            print('Jaad dataset does not have speed information')
            print('Vehicle actions are used instead')
        if balance:
            self.balance_data_samples(d, data_raw['image_dimension'][0])
        d['box_org'] = d['box'].copy()
        d['tte'] = []

        if isinstance(time_to_event, int):
            for k in d.keys():
                for i in range(len(d[k])):
                    d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
            d['tte'] = [[time_to_event]] * len(data_raw['bbox'])
        else:
            overlap = opts['overlap']  # if data_type == 'train' else 0.0
            olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
            olap_res = 1 if olap_res < 1 else olap_res
            for k in d.keys():
                seqs = []
                for seq in d[k]:
                    start_idx = len(seq) - obs_length - time_to_event[1]
                    end_idx = len(seq) - obs_length - time_to_event[0]
                    seqs.extend([seq[i:i + obs_length] for i in
                                 range(start_idx, end_idx + 1, olap_res)])
                d[k] = seqs

            for seq in data_raw['bbox']:
                start_idx = len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - time_to_event[0]
                d['tte'].extend([[len(seq) - (i + obs_length)] for i in
                                 range(start_idx, end_idx + 1, olap_res)])
        if normalize:
            for k in d.keys():
                if k != 'tte':
                    if k != 'box' and k != 'center':
                        for i in range(len(d[k])):
                            d[k][i] = d[k][i][1:]
                    else:
                        for i in range(len(d[k])):
                            d[k][i] = np.subtract(d[k][i][1:], d[k][i][0]).tolist()
                d[k] = np.array(d[k])
        else:
            for k in d.keys():
                d[k] = np.array(d[k])

        d['crossing'] = np.array(d['crossing'])[:, 0, :]
        pos_count = np.count_nonzero(d['crossing'])
        neg_count = len(d['crossing']) - pos_count
        print("Negative {} and positive {} sample counts".format(neg_count, pos_count))

        return d, neg_count, pos_count
    def get_data(self, data_type, data_raw, model_opts,save_root_folder):
        assert model_opts['obs_length'] == 16
        model_opts['normalize_boxes'] = False
        self._generator = model_opts.get('generator', False)
        data_type_sizes_dict = {}
        dataset = model_opts['dataset']
        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        data_type_sizes_dict['box'] = data['box'].shape[1:]
        if 'speed' in data.keys():
            data_type_sizes_dict['speed'] = data['speed'].shape[1:]
        _data = []

        for d_type in model_opts['obs_input_type']:
            if 'scene image' == d_type:
                path_to_scene, _ = get_path(save_folder='scene image',
                                           dataset=dataset,
                                           save_root_folder=save_root_folder)

                self.create_scene_image(data['image'],
                                                data['ped_id'],
                                                data['box'],
                                                data_type=data_type,
                                                file_path= path_to_scene,
                                                dataset=model_opts['dataset'])
            elif 'scene image pedestrain bounding box' == d_type:
                path_to_scene_pedestrain, _ = get_path(save_folder='scene image pedestrain bounding box',
                                           dataset=dataset,
                                           save_root_folder=save_root_folder)
                self.create_scene_image_pedestrain(data['image'],
                                                    data['ped_id'],
                                                    data['box'],
                                                    data_type=data_type,
                                                    file_path=path_to_scene_pedestrain,
                                                    dataset=model_opts['dataset'])
            elif 'pedestrian image' == d_type:
                path_to_pedestrain_image, _ = get_path(save_folder='pedestrian image',dataset=dataset,save_root_folder=save_root_folder)
                self.create_pedestrain_image(data['image'],
                                                    data['ped_id'],
                                                    data['box'],
                                                    data_type=data_type,
                                                    file_path=path_to_pedestrain_image,
                                                    dataset=model_opts['dataset'])
            elif 'pedestrain pose' == d_type:
                path_to_pedestrain_pose,_=get_path(save_folder='pdestrain pose',dataset=dataset,save_root_folder=save_root_folder)
                self.creat_pose(data['image'],
                                                    data['ped_id'],
                                                    data['box'],
                                                    data_type=data_type,
                                                    file_path=path_to_pedestrain_pose,
                                                    dataset=model_opts['dataset'])
            elif 'pedestrian box' == d_type:
                path_to_pedestrain_box,_=get_path(save_folder='pedestrain box',dataset=dataset,save_root_folder=save_root_folder)
                self.creat_box(data['image'],
                                                    data['ped_id'],
                                                    data['box'],
                                                    data_type=data_type,
                                                    file_path=path_to_pedestrain_box,
                                                    dataset=model_opts['dataset'])
            elif 'label' == d_type:
                path_to_label,_=get_path(save_folder='label',dataset=dataset,save_root_folder=save_root_folder)
                self.creat_label(data['image'],
                                              data['ped_id'],
                                              data['box'],
                                              data['crossing'],
                                              data_type=data_type,
                                              file_path=path_to_label,
                                              dataset=model_opts['dataset'])
            elif 'speed' == d_type:
                path_to_speed,_=get_path(save_folder='speed',dataset=dataset,save_root_folder=save_root_folder)
                self.creat_speed(data['image'],
                                              data['ped_id'],
                                              data['box'],
                                              data['speed'],
                                              data_type=data_type,
                                              file_path=path_to_speed,dataset=model_opts['dataset']
                )

            else:
                print('unknown data type')
        print('数据生成完成！')


