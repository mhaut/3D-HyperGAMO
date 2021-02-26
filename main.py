import os
import auxil
import argparse
import numpy as np
import dense_suppli as spp
import dense_net3D_SA as nt
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Input
from keras.models import Model



def load_hyper(args):
    if args.dataset not in ["UH", "DIP", "DUP"]:
        acrstate = None if args.random_state == None else args.random_state+args.idtest
        pixelsO, labelsO, numclass = auxil.loadData(args.dataset, num_components=args.components, rand_state=acrstate)
        pixelsO, labelsO = auxil.createImageCubes(pixelsO, labelsO, windowSize=args.spatialsize, removeZeroLabels = False)
        pixels = pixelsO[labelsO!=0]
        labels = labelsO[labelsO!=0] - 1
        bands = pixels.shape[-1]; numberofclass = len(np.unique(labels))
        pixels = pixels.reshape((pixels.shape[0], args.spatialsize, args.spatialsize, bands, 1))
        pixelsO = pixelsO.reshape((pixelsO.shape[0], args.spatialsize, args.spatialsize, bands, 1))
        x_train, x_test, y_train, y_test = auxil.split_data(pixels, labels, args.tr_percent, rand_state=acrstate)
        del pixels, labels
    else:
        x_train, x_test, y_train, y_test, pixelsO, labelsO, bands, numberofclass = auxil.loadDataFIX(args)
    return (x_train, y_train), (x_test, y_test), (pixelsO, labelsO), numberofclass, bands





def main():
    parser = argparse.ArgumentParser(description='PyTorch DCNNs Training')
    parser.add_argument('--max_step', default=8000, type=int, help='number of total epochs to run')
    parser.add_argument('--idtest', default=0, type=int, help='id of experiment')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--components', default=None, type=int, help='dimensionality reduction')
    parser.add_argument('--dataset', default='IP', type=str, help='dataset (options: IP, UP, SV, KSC)')
    parser.add_argument('--tr_percent', default=0.10, type=float, help='samples of train set')
    parser.add_argument('--tr_bsize', default=100, type=int, help='mini-batch train size (default: 100)')
    parser.add_argument('--inplanes', dest='inplanes', default=16, type=int, help='bands before blocks')
    parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false', help='to use basicblock (default: bottleneck)')
    parser.add_argument('--spatialsize', dest='spatialsize', default=7, type=int, help='spatial-spectral patch dimension')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--random_state', default=None, type=int, help='random seed')

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}
    
    #args.components = 35 if args.dataset == 'IP' else 15
    (trainS, labelTr), (testS, labelTs), (pixels, labels), numberofclass, bands = load_hyper(args)

    
    n, m = trainS.shape[0], testS.shape[0]
    labelTr, labelTs, c, pInClass, _, newOrder = spp.relabel(labelTr, labelTs)
    imbalancedCls, toBalance, imbClsNum, ir=spp.irFind(pInClass, c)
    
    labelsCat = to_categorical(labelTr)

    shuffleIndex=np.random.choice(np.arange(n), size=(n,), replace=False)
    trainS=trainS[shuffleIndex]
    labelTr=labelTr[shuffleIndex]
    labelsCat=labelsCat[shuffleIndex]
    classMap=list()
    for i in range(c):
        classMap.append(np.where(labelTr==i)[0])

    
    adamOpt=Adam(0.0002, 0.5)
    latDim, modelSamplePd, resSamplePd=100, 8000, 500
    
    # model initialization
    mlp=nt.denseMlpCreate(sh=(args.spatialsize, args.spatialsize, bands, 1, ), num_class=numberofclass)
    mlp.compile(loss='mean_squared_error', optimizer=adamOpt)
    mlp.trainable=False

    dis=nt.denseDisCreate((args.spatialsize, args.spatialsize, bands, 1, ), num_class=numberofclass)
    dis.compile(loss='mean_squared_error', optimizer=adamOpt)
    dis.trainable=False

    gen=nt.denseGamoGenCreate(latDim, numberofclass)


    gen_processed, genP_mlp, genP_dis=list(), list(), list()
    for i in range(imbClsNum):
        dataMinor=trainS[classMap[i], :]
        numMinor=dataMinor.shape[0]
        gen_processed.append(nt.denseGenProcessCreate(numMinor, dataMinor,sh = \
            (args.spatialsize, args.spatialsize, bands, 1),mul = args.spatialsize * args.spatialsize * bands))

        ip1=Input(shape=(latDim,))
        ip2=Input(shape=(c,))
        op1=gen([ip1, ip2])
        op2=gen_processed[i](op1)
        op3=mlp(op2)
        genP_mlp.append(Model(inputs=[ip1, ip2], outputs=op3))
        genP_mlp[i].compile(loss='mean_squared_error', optimizer=adamOpt)

        ip1=Input(shape=(latDim,))
        ip2=Input(shape=(c,))
        ip3=Input(shape=(c,))
        op1=gen([ip1, ip2])
        op2=gen_processed[i](op1)
        op3=dis([op2, ip3])
        genP_dis.append(Model(inputs=[ip1, ip2, ip3], outputs=op3))
        genP_dis[i].compile(loss='mean_squared_error', optimizer=adamOpt)


    batchDiv, numBatches, bSStore = spp.batchDivision(n, args.tr_bsize)
    genClassPoints=int(np.ceil(args.tr_bsize / c))

    fileStart = './SavedModel/'
    savePath = './SavedModel/'
    fileEnd = '_Model.h5'
    if not os.path.exists(fileStart):
        os.makedirs(fileStart)
    picPath=savePath+'Pictures'
    if not os.path.exists(picPath):
        os.makedirs(picPath)


    iter=np.int(np.ceil(args.max_step/resSamplePd)+1)
    acsaSaveTr, gmSaveTr, accSaveTr=np.zeros((iter,)), np.zeros((iter,)), np.zeros((iter,))
    acsaSaveTs, gmSaveTs, accSaveTs=np.zeros((iter,)), np.zeros((iter,)), np.zeros((iter,))
    confMatSaveTr, confMatSaveTs=np.zeros((iter, c, c)), np.zeros((iter, c, c))
    tprSaveTr, tprSaveTs=np.zeros((iter, c)), np.zeros((iter, c))



    step=0
    bestacc = -1
    while step < args.max_step:
        for j in range(numBatches):
            x1, x2=batchDiv[j, 0], batchDiv[j+1, 0]
            validR=np.ones((bSStore[j, 0],1))-np.random.uniform(0,0.1, size=(bSStore[j, 0], 1))
            mlp.train_on_batch(trainS[x1:x2], labelsCat[x1:x2])
            dis.train_on_batch([trainS[x1:x2], labelsCat[x1:x2]], validR)

            invalid=np.zeros((bSStore[j, 0], 1))+np.random.uniform(0, 0.1, size=(bSStore[j, 0], 1))
            randNoise=np.random.normal(0, 1, (bSStore[j, 0], latDim))
            fakeLabel=spp.randomLabelGen(toBalance, bSStore[j, 0], c)
            rLPerClass=spp.rearrange(fakeLabel, imbClsNum)
            fakePoints=np.zeros((bSStore[j, 0],args.spatialsize, args.spatialsize, bands, 1))
            genFinal=gen.predict([randNoise, fakeLabel])
            for i1 in range(imbClsNum):
                if rLPerClass[i1].shape[0]!=0:
                    temp=genFinal[rLPerClass[i1]]
                    fakePoints[rLPerClass[i1]]=gen_processed[i1].predict(temp)

            mlp.train_on_batch(fakePoints, fakeLabel)
            dis.train_on_batch([fakePoints, fakeLabel], invalid)

            for i1 in range(imbClsNum):
                validA=np.ones((genClassPoints, 1))
                randomLabel=np.zeros((genClassPoints, c))
                randomLabel[:, i1]=1
                randNoise=np.random.normal(0, 1, (genClassPoints, latDim))
                oppositeLabel=np.ones((genClassPoints, c))-randomLabel
                genP_mlp[i1].train_on_batch([randNoise, randomLabel], oppositeLabel)
                genP_dis[i1].train_on_batch([randNoise, randomLabel, randomLabel], validA)

            if step%resSamplePd==0:
                saveStep=int(step//resSamplePd)

                pLabel=np.argmax(mlp.predict(trainS), axis=1)
                acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTr)
                print('Train: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
                print('TPR: ', np.round(tpr, 2))
                acsaSaveTr[saveStep], gmSaveTr[saveStep], accSaveTr[saveStep]=acsa, gm, acc
                confMatSaveTr[saveStep]=confMat
                tprSaveTr[saveStep]=tpr

                pLabel=np.argmax(mlp.predict(testS), axis=1)
                acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTs)
                print('Test: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
                print('TPR: ', np.round(tpr, 2))
                acsaSaveTs[saveStep], gmSaveTs[saveStep], accSaveTs[saveStep]=acsa, gm, acc
                confMatSaveTs[saveStep]=confMat
                tprSaveTs[saveStep]=tpr
                
                results = auxil.reports(pLabel, labelTs)[2]
                if bestacc <= results[0]:
                    bestacc = auxil.reports(pLabel, labelTs)[2][0]
                    resultsF = auxil.reports(pLabel, labelTs)[2]

            if step%modelSamplePd==0 and step!=0:
                direcPath=savePath+'gamo_models_'+str(step)
                if not os.path.exists(direcPath):
                    os.makedirs(direcPath)
                gen.save(direcPath+'/GEN_'+str(step)+fileEnd)
                mlp.save(direcPath+'/MLP_'+str(step)+fileEnd)
                dis.save(direcPath+'/DIS_'+str(step)+fileEnd)
                for i in range(imbClsNum):
                    gen_processed[i].save(direcPath+'/GenP_'+str(i)+'_'+str(step)+fileEnd)

            step=step+2
            if step>=args.max_step: break

    print(newOrder)
    print(resultsF)

if __name__ == '__main__':
	main()
