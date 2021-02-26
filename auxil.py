import os
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import copy

def random_unison(a,b, rstate=None):
    assert len(a) == len(b)
    p = np.random.RandomState(seed=rstate).permutation(len(a))
    return a[p], b[p]


def loadDataFIX(args):
    data, labels, numclass = loadData(args.dataset, num_components=args.components)
    num_bands = data.shape[-1]
    pixels, labels = createImageCubes(data, labels, windowSize=args.spatialsize, removeZeroLabels = False)
    num_class = len(np.unique(labels)) - 1
    if args.dataset == "UH":
        data_path = os.path.join(os.getcwd(),'HSI-datasets/Classification/')
        y_train = sio.loadmat(os.path.join(data_path, 'houston_gt.mat'))['houston_gt_tr'].reshape(-1)
        y_test = sio.loadmat(os.path.join(data_path, 'houston_gt.mat'))['houston_gt_te'].reshape(-1)
        Xtrain = pixels[y_train!=0,:,:,:]
        Xtest  = pixels[y_test!=0,:,:,:]
        del pixels
        y_train = y_train[y_train!=0] - 1
        y_test  = y_test[y_test!=0] - 1
        return Xtrain, Xtest, y_train, y_test, num_bands, num_class
    elif args.dataset == "DIP":
        y_train2 = sio.loadmat(os.path.join('HSI-datasets/Classification/', 'indianpines_disjoint_dset.mat'))\
                                            ['indianpines_disjoint_dset']
        y_test = sio.loadmat(os.path.join('HSI-datasets/Classification/', 'indian_pines_corrected_gt.mat'))['indian_pines_gt']
        y_train = copy.deepcopy(y_train2)
        for i, val in enumerate([0,2,3,5,6,8,10,11,12,14,1,4,7,9,13,15,16]): y_train[y_train2==i] = val
        del y_train2
        y_test[y_train!=0] = 0
    elif args.dataset == "DUP":
        y_train = sio.loadmat(os.path.join('HSI-datasets/Classification/', 'TRpavia_fixed.mat'))['TRpavia_fixed'].reshape(-1)
        y_test = sio.loadmat(os.path.join('HSI-datasets/Classification/', 'TSpavia_fixed.mat'))['TSpavia_fixed'].reshape(-1)
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    Xtrain = pixels[y_train!=0,:]
    Xtest  = pixels[y_test!=0,:]
    #del pixels
    y_train = y_train[y_train!=0] - 1
    y_test  = y_test[y_test!=0] - 1
    Xtrain, y_train = random_unison(Xtrain,y_train, rstate=None)
    return Xtrain, Xtest, y_train, y_test, pixels, labels, num_bands, num_class


def split_data(pixels, labels, percent, splitdset="sklearn", with_val=False, rand_state=None):
    return train_test_split(pixels, labels, test_size=(1-percent), stratify=labels, random_state=rand_state)

def loadData(name, num_components=None, rand_state=None):
    data_path = os.path.join(os.getcwd(),'HSI-datasets/Classification/')
    if name in ['IP', 'DIP']:
        data = sio.loadmat(os.path.join(data_path, 'indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'indian_pines_corrected_gt.mat'))['indian_pines_gt']
    elif name == 'SV':
        data = sio.loadmat(os.path.join(data_path, 'salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'salinas_gt.mat'))['salinas_gt']
    elif name in ['UP', 'DUP']:
        data = sio.loadmat(os.path.join(data_path, 'paviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'paviaU_gt.mat'))['paviaU_gt']
    elif name == 'KSC':
        data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
    elif name == 'UH':
        data = sio.loadmat(os.path.join(data_path, 'houston.mat'))['houston']
        labels = sio.loadmat(os.path.join(data_path, 'houston_gt.mat'))['houston_gt_tr']
        labels += sio.loadmat(os.path.join(data_path, 'houston_gt.mat'))['houston_gt_te']
    elif name == 'BW':
        data = sio.loadmat(os.path.join(data_path, 'Botswana.mat'))['Botswana']
        labels = sio.loadmat(os.path.join(data_path, 'Botswana_gt.mat'))['Botswana_gt']
    else:
        print("NO DATASET")
        exit()

    shapeor = data.shape
    data = data.reshape(-1, data.shape[-1])
    if num_components != None:
        data = PCA(n_components=num_components, random_state=rand_state).fit_transform(data)
        shapeor = np.array(shapeor)
        shapeor[-1] = num_components
    data = MinMaxScaler(feature_range=(-1, 1)).fit_transform(data)
    data = data.reshape(shapeor)
    num_class = len(np.unique(labels)) - 1
    return data, labels, num_class


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2])).astype("float32")
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch.astype("float32")
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels.astype("int")


def accuracy(output, target, topk=(1,)):
    output = np.argmax(output, axis=1)
    return np.sum(output == target) * 100


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(y_pred, y_test):
    classification = classification_report(y_test, y_pred)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)

    return classification, confusion, list(np.round(np.array([oa, aa, kappa] + list(each_acc)) * 100, 2))
