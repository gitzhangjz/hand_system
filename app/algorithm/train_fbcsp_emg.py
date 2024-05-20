import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt
from impl.bin.MLEngine import FilterBank
from impl.bin.FBCSP import FBCSP
from sklearn.svm import SVR
from impl.bin.Classifier import Classifier
import os
def save_data(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def read_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

'''
给信号降采样, pre_fq为原始采样率
data(chan_n, sample_n)
'''
def resample(data, pre_fq , fq = 250):
    info = mne.create_info(
        ch_names=list(str(i) for i in range(data.shape[0])),
        ch_types="eeg",  # channel type
        sfreq = pre_fq,  # frequency
    )
    raw = mne.io.RawArray(data, info)  # create raw
    # raw.notch_filter(freqs=50, method='spectrum_fit', filter_length='auto', phase='zero-double', verbose=True)
    raw.resample(fq)
    # raw.filter(f_low, f_high)
    return raw.get_data()

'''
给脑电滤波, data(channel_n*sample_n)
针对一个样本滤波
'''
def _filter_process(data, f_low, f_high, fq = 250):
    info = mne.create_info(
        ch_names=list(str(i) for i in range(data.shape[0])),
        ch_types="eeg",  # channel type
        sfreq = fq,  # frequency
    )
    raw = mne.io.RawArray(data, info)  # create raw
    raw.filter(f_low, f_high)
    return raw.get_data()


'''
给脑电滤波, data(n * channel_n * sample_n)
分别对每个样本滤波
'''
def filter_process(data, low_fq, high_fq):
    for i in range(len(data)):
        data[i] = _filter_process(data[i], low_fq, high_fq)
    return data

'''
画折线图
'''
def draw(data, x=1,y=1,i=1):
    plt.subplot(x,y,i)
    plt.plot(data)

''' 
把信号拉直data(sample_n)
'''
def stretch(data:np.array) -> np.array:
    t = 10
    sum = np.sum(data[:t])
    ret = [ data[i]-sum/t for i in range(10)]
    for i in range(10, data.shape[0]):
        sum += data[i]
        sum -= data[i-10]
        ret.append(data[i]-(sum/t))
    return np.array(ret)

'''
把前后2l长度的值平均,小于threshold(无用信号)的归零,其他的等于信号的绝对值
'''
def emg_filter(data, threshold, l = 5):
    ret = np.zeros_like(data)
    for i in range(l, data.shape[0]-l):
        x = np.sum(np.abs(data[i-l:i+l]))/(2*l)
        ret[i] = np.abs(data[i]) if x > threshold else 0
    return ret

'''
data(n*2*500)
对两个通道拉直后做阈值过滤
'''
def emg_pre_process(data, ch_0_threshold = 8, ch_1_threshold = 30, l = 5):
    for i in range(data.shape[0]):
        data[i,0] = emg_filter(stretch(data[i,0]), ch_0_threshold)
        data[i,1] = emg_filter(stretch(data[i,1]), ch_0_threshold)
    return data

'''
data(channel_n * sample_n)
提取通道1、2的: 均值，极差, 以及两通道均值比
返回 (1*5), 5个特征
'''
def _emg_feature(data):
    a = np.zeros(5)
    channel1 = data[0,:]
    channel2 = data[1,:]
    a[0] = np.mean(channel1)
    a[1] = np.mean(channel2)
    a[2] = np.mean(channel1)/np.mean(channel2)
    a[3] = np.max(channel1)-np.min(channel1)
    a[4] = np.max(channel2)-np.min(channel2)
    return a
'''
data(n * channel_n * sample_n)
提取通道1、2的: 均值，极差, 以及两通道均值比
返回 (n*5), 每个样本5个特征
'''
def emg_feature(data):
    ret = np.zeros((0,5))
    for i in range(data.shape[0]):
        a = _emg_feature(data[i])
        ret = np.vstack([ret,a])
    return ret

def get_multi_class_regressed(y_predicted):
    y_predict_multi = np.asarray([np.argmin(y_predicted[i,:]) for i in range(y_predicted.shape[0])])
    return y_predict_multi

'''
生成k组indx
'''
def cross_validate_sequential_split(y_labels, k_fold):
    from sklearn.model_selection import StratifiedKFold
    train_indices = {}
    test_indices = {}
    skf_model = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=2023)
    i = 0
    for train_idx, test_idx in skf_model.split(np.zeros(len(y_labels)), y_labels):
        train_indices.update({i: train_idx})
        test_indices.update({i: test_idx})
        i += 1
    return train_indices, test_indices

def Train(data, label, low_fq, high_fq, eeg_channels, emg_channels, name):
    model_path = os.path.join(os.getcwd(), 'models')
    y_train = label

    # 通道选择，data的前24通道是eeg，后两个通道是emg
    eeg_data = data[:,eeg_channels,:]
    emg_data = data[:,emg_channels,:]

    # eeg 滤波
    eeg_data = filter_process(eeg_data, low_fq, high_fq)
    # 
    emg_data = emg_pre_process(emg_data)

    # 对eeg多频带滤波
    fbank = FilterBank(fs=250)
    fbank.get_filter_coeff()
    save_data(fbank, os.path.join(model_path,'FilterBank.pkl'))
    x_train_fb = fbank.filter_data(eeg_data)
    m_filters = 2

    y_classes_unique = np.array([0,1,2])
    n_classes = len(np.unique(y_train))

    fbcsp = FBCSP(m_filters)
    fbcsp.fit(x_train_fb,y_train)
    save_data(fbcsp, os.path.join(model_path, name+'_fbcsp.pkl'))

    y_train_predicted = np.zeros((y_train.shape[0], n_classes), dtype=float)

    # emg特征提取
    emg_train = emg_feature(emg_data)
    
    for j in range(n_classes):
        cls_of_interest = y_classes_unique[j]
        select_class_labels = lambda cls, y_labels: [0 if y == cls else 1 for y in y_labels]

        # 将y=[1,2,2,1,3,...] 转化成 y = [0,1,1,0,1,..]格式
        y_train_cls = np.asarray(select_class_labels(cls_of_interest, y_train))

        # 提取FBCSP特征 单用脑电
        x_features_train = fbcsp.transform(x_train_fb, class_idx=cls_of_interest)
        # print(x_features_train[1])
        # 特征拼接
        x_features_train = np.hstack([x_features_train, emg_train])
        # print(x_features_train[0])
        classifier_type = SVR(gamma='auto')
        classifier = Classifier(classifier_type)

        # 将FBCSP特征作为x, [0,1,1,0,..]作为y，训练
        y_train_predicted[:,j] = classifier.fit(x_features_train, np.asarray(y_train_cls,dtype=float))
        save_data(classifier, os.path.join(model_path, name+'_class_'+str(j)+'.pkl'))
        # print(read_data('./models/zjz_class_'+str(j)+'.pkl').predict(x_features_train[0:1]))

    # print(x_features_train[0])
    print(y_train_predicted[1])

    
    y_train_predicted_multi = get_multi_class_regressed(y_train_predicted)

    tr_acc =np.sum(y_train_predicted_multi == y_train, dtype=float) / len(y_train)
    print(f'Training Accuracy = {str(tr_acc)}')
    return tr_acc
    
def verify(data, label, low_fq, high_fq, eeg_channels, emg_channels, k_fold = 5):
    # 通道选择，data的前24通道是eeg，后两个通道是emg
    eeg_data = data[:,eeg_channels,:]
    emg_data = data[:,emg_channels,:]

    # eeg 滤波
    eeg_data = filter_process(eeg_data, low_fq, high_fq)

    # 
    emg_data = emg_pre_process(emg_data)
    # print("eeg : ",eeg_data.shape) #(300, 3, 500)
    # print("emg : ",emg_data.shape) # emg :  (300, 2, 500)
    # 对eeg多频带滤波
    fbank = FilterBank(fs=250)
    fbank.get_filter_coeff()
    data_4d = fbank.filter_data(eeg_data)
    # print("data_4d : ",data_4d.shape) # (9, 300, 3, 500)
    
    m_filters = 2
    training_accuracy = []
    testing_accuracy = []

    # 生成K_fold组indxes
    train_indxes, test_indxes = cross_validate_sequential_split(label, k_fold)
    for i in range(k_fold):
        train_idx = train_indxes.get(i)
        test_idx = test_indxes.get(i)

        y_train, y_test = label[train_idx], label[test_idx]
        x_train_fb, x_test_fb = data_4d[:,train_idx,:], data_4d[:,test_idx,:]
        emg_train, emg_test = emg_feature(emg_data[train_idx]), emg_feature(emg_data[test_idx])
        # print("emg_train : ",emg_train.shape) # (train_n, 1)
        # print("emg_test : ",emg_test.shape) # (test_n, 1)

        y_classes_unique = np.unique(y_train)
        n_classes = len(np.unique(y_train))

        fbcsp = FBCSP(m_filters)
        fbcsp.fit(x_train_fb,y_train)
        y_train_predicted = np.zeros((y_train.shape[0], n_classes), dtype=float)
        y_test_predicted = np.zeros((y_test.shape[0], n_classes), dtype=float)

        # 对每个类做二分类, 每个类会predicte一个小数, 最小值所属的类为预测的类
        for j in range(n_classes):
            cls_of_interest = y_classes_unique[j]
            select_class_labels = lambda cls, y_labels: [0 if y == cls else 1 for y in y_labels]

            # 将y=[1,2,2,1,3,...] 转化成 y = [0,1,1,0,1,..]格式
            y_train_cls = np.asarray(select_class_labels(cls_of_interest, y_train))
            y_test_cls = np.asarray(select_class_labels(cls_of_interest, y_test))

            # 提取FBCSP特征 单用脑电
            x_features_train = fbcsp.transform(x_train_fb, class_idx=cls_of_interest)
            x_features_test = fbcsp.transform(x_test_fb, class_idx=cls_of_interest)
            
            # 单用肌电
            # x_features_train = emg_train
            # x_features_test = emg_test

            # 将EEG特征和EMG特征拼接 特征拼接
            # x_features_train = np.hstack([x_features_train, emg_train])
            # x_features_test = np.hstack([x_features_test, emg_test])
            
            # x_features_train = np.nan_to_num(x_features_train,posinf=10.,neginf=-10.)
            # x_features_test = np.nan_to_num(x_features_test,posinf=10.,neginf=-10.)
            # print("feature", np.isinf(x_features_train).any(), np.isfinite(x_features_train).all(), np.isnan(x_features_train).any())
            # print("x_features_train : ",x_features_train.shape) # (train_n, 41)
            # print("x_features_test : ",x_features_test.shape) # (test_n, 41)

            classifier_type = SVR(gamma='auto')
            classifier = Classifier(classifier_type)

            # 将FBCSP特征作为x, [0,1,1,0,..]作为y，训练
            y_train_predicted[:,j] = classifier.fit(x_features_train, np.asarray(y_train_cls,dtype=float))
            y_test_predicted[:,j] = classifier.predict(x_features_test)

        y_train_predicted_multi = get_multi_class_regressed(y_train_predicted)
        y_test_predicted_multi = get_multi_class_regressed(y_test_predicted)

        tr_acc =np.sum(y_train_predicted_multi == y_train, dtype=float) / len(y_train)
        te_acc =np.sum(y_test_predicted_multi == y_test, dtype=float) / len(y_test)


        print(f'Training Accuracy = {str(tr_acc)}')
        print(f'Testing Accuracy = {str(te_acc)}\n')

        training_accuracy.append(tr_acc)
        testing_accuracy.append(te_acc)

    mean_training_accuracy = np.mean(np.asarray(training_accuracy))
    mean_testing_accuracy = np.mean(np.asarray(testing_accuracy))

    print('*'*10)
    print(f'Mean Training Accuracy = {str(mean_training_accuracy)}')
    print(f'Mean Testing Accuracy = {str(mean_testing_accuracy)}')
    print('*' * 10)
    return mean_training_accuracy, mean_testing_accuracy


'''
    将数据处理成(n * chan_n * sample_n)
    label是(n*1)
'''
def train_preprocess(uid:str):
    mne.set_log_level(verbose="ERROR")
    root_path = os.path.join(os.getcwd(), 'app','data',uid)
    eeg_fq = 300
    emg_fq = 1000
    # 目标降采样频率
    fq = 250
    # eeg(24)+emg(2)通道数
    channel_n = 26

    start = 4*fq
    end = 6*fq
    sample_n = end-start

    data = np.empty((0, channel_n, sample_n))
    labels = []

    for i in range(1,11):
        if not os.path.exists(os.path.join(root_path,f'eeg_{i}.pkl')):
            break
        print("preprocess: "+str(i))
        eeg_data = read_data(os.path.join(root_path,f'eeg_{i}.pkl'))
        emg_data = read_data(os.path.join(root_path,f'emg_{i}.pkl'))
        eeg_triger = read_data(os.path.join(root_path,f'eeg_triger_{i}.pkl'))
        emg_triger = read_data(os.path.join(root_path,f'emg_triger_{i}.pkl'))
        label = read_data(os.path.join(root_path,f'label{i}.pkl'))

        # 降采样处理
        eeg_data = resample(eeg_data, eeg_fq)
        emg_data = resample(emg_data, emg_fq)
        eeg_triger = np.array([int(i/eeg_fq*fq) for i in eeg_triger])
        emg_triger = np.array([int(i/emg_fq*fq) for i in emg_triger])
        for i in range(30):
            eeg_sample = eeg_data[:24, eeg_triger[i]+start : eeg_triger[i]+end]
            emg_sample = emg_data[:2, emg_triger[i]+start : emg_triger[i]+end]
            data =  np.vstack([data, np.vstack([eeg_sample, emg_sample])[np.newaxis,:]])
            labels.append(int(label[i]))
    save_data({'data':data, 'labels':np.array(labels)}, os.path.join(root_path, uid+'_data.pkl'))


'''
    接受一个(1, 26, 2*fq)的数据, 只降采样到fq
    返回分类结果(0休息,1抓握,2伸展)
'''
def predict(data, name):
    eeg_chan = [1,5,7]
    emg_chan = [-2,-1]

    eeg_data = data[:, eeg_chan, :]
    emg_data = data[:, emg_chan, :]

     # eeg 滤波
    eeg_data = filter_process(eeg_data, 8, 12)
    # emg预处理
    emg_data = emg_pre_process(emg_data)

    model_path =  os.path.join(os.getcwd(), 'models')
    if not os.path.exists(os.path.join(model_path,name+'_fbcsp.pkl')):
        name = '9'
    fbank = read_data(os.path.join(model_path,'FilterBank.pkl'))
    fbcsp = read_data(os.path.join(model_path,name+'_fbcsp.pkl'))
    classifiers = []
    for i in range(3):
        classifiers.append(read_data(os.path.join(model_path,name+'_class_'+str(i)+'.pkl')))

    eeg_data = fbank.filter_data(eeg_data)

    y_classes_unique = np.array([0,1,2])
    n_classes = 3
    emg_features = emg_feature(emg_data)

    y = np.zeros((1,3))
    for i in range(n_classes):
        cls_of_interest = y_classes_unique[i]

        # 提取FBCSP特征 单用脑电
        eeg_features = fbcsp.transform(eeg_data, class_idx=cls_of_interest)
        # 将EEG特征和EMG特征拼接 特征拼接
        feature = np.hstack([eeg_features, emg_features])

        classifier = classifiers[i]

        y[:,i] = classifier.predict(feature)
    return y.argmin()

'''
    eeg(25,n)
    emg(9,m)
'''
def preprocess(eeg_data, emg_data) :
    mne.set_log_level(verbose="ERROR")
    eeg_fq = 300
    emg_fq = 1000
    # 目标降采样频率
    fq = 250
    # eeg(24)+emg(2)通道数
    channel_n = 26

    # 取后2s
    sample_n = 2*fq

    # 降采样处理
    eeg_data = resample(eeg_data, eeg_fq)[:24,-sample_n:]
    emg_data = resample(emg_data, emg_fq)[:2, -sample_n:]
    # print("eeg_shape", eeg_data.shape)
    # print("emg_shape", emg_data.shape)
    data = np.vstack([eeg_data, emg_data])[np.newaxis,:]
    return data


if __name__ == "__main__":
    train_preprocess("10")
#     # train_preprocess()
#     # 取消mne打印日志
#     # mne.set_log_level(verbose="ERROR")
#     # root_path = './data/good_data/'
#     # data = d2l.read_data(root_path+'zjz_data.pkl')
#     # print(data['data'].shape)

#     # verify(data['data'][:],data['labels'][:], 8, 12, [1,5,7], [-2,-1])
    
#     # train(data['data'][:],data['labels'][:], 8, 12, [1,5,7], [-2,-1], 'zjz')


#     # for i in range(0, len(data['data'])):
#     #     res = predict(data['data'][i:i+1], 'zjz')

#     root_path = './app/algorithm/data/good_data/'
#     eeg_data = read_data(root_path+f'eeg_1.pkl')
#     emg_data = read_data(root_path+f'emg_1.pkl')

#     data = preprocess(eeg_data, emg_data)
#     print(predict(data, 'zjz'))
