import numpy as np
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.morphology import dilation, square
from skimage import io
import os
import tensorflow as tf
from keras.optimizers import Adam
from dlia_tools.u_net import u_net
from dlia_tools.keras_custom_loss import jaccard2_loss
import scipy.io
from Traincenterv2 import readweaving
import keras
import matplotlib.pyplot as plt
import cv2
from tifffile import imwrite

def writeNpArray2tif(a, tiffFileName):
    '''
    npArray2tif(a, tiffFileName)
    a : 3D numpy array
    tiffFileName : name of the output file, as a string
    
    '''
    D, H, W = a.shape
    a = (a * 255).round().astype(np.uint8) # (pour des images 8 bits)
    a.shape = 1, D, 1, H, W, 1
    imwrite(tiffFileName, a, imagej=True)

def NoiseCleaning(y, point, seuil, gt, metric = "euclidean"):
    """
    Arguments:
        y:list of point
        point:objects to clean
    Returns:
        cleaned list y
    """
    d2y =cdist(y, point, metric=metric)
    idx_y = np.argmin(d2y, axis=0)
    miny = np.amin(d2y, axis=0)
    if ((point[0][0]<=max(gt[:,0])+10)and(point[0][1]<=max(gt[:,1])+10))and(((point[0][0]>=min(gt[:,0])-10)and(point[0][1]>=min(gt[:,1])-10))):
        if(miny<seuil):
            npoint= np.mean([y[idx_y][0],point[0]], axis=0)
            y=list(y)
            del y[idx_y[0]]
            y.append(npoint)
        else:
            y=list(y)
            y.append(point[0])
        y=np.array(y)
    return y
def result(pred,I):
    M=np.zeros(I.shape,dtype=np.uint8)
    img=(pred * 255).round().astype(np.uint8)
    #ret,thresh = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
    thresh=img>20
    label_img = label(thresh)
    regions = regionprops(label_img)
    for props in regions:
        #print('Area',props.area)
        if props.area>20:
            x,y=props.centroid
            y=int(y)
            x=int(x)
            M[x,y]=1
    Slice=((1-dilation(M>0,square(11)))*I)
    return Slice

def PostProcessing(pred, verite):
    """
    Arguments:
        pred: numpy array representing the image of the prediction
        verite: : numpy array representing the image of the ground truth
    Returns:
        Y: distance matrix between all points from the ground truth and the prediction
        gt: list of the coordinate of the ground truth
        centroid: : list of the coordinate of the prediction
    """
    gt=np.transpose(np.nonzero(verite))
    gray = cv2.cvtColor(pred,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,30,255,cv2.THRESH_OTSU)
    label_img = label(thresh)
    regions = regionprops(label_img)
    centroid=[]
    token=1
    for props in regions:
        if props.area>20:
            x,y=props.centroid
            y=int(y)
            x=int(x)
            if token:
                centroid.append(np.array([x,y]))
                centroid=np.array(centroid)
                token=0
            else:
                centroid=NoiseCleaning(centroid, np.array([np.array([x,y])]), 50, gt, metric = "euclidean")
    Y = cdist(centroid, gt, 'euclidean')
    return Y,gt,centroid


def Evaluation(x):
    """
    Arguments:
        x: the distance matrix of the prediction to evaluate
    Returns:
        rate1: sensitivity
        rate2: precision
    """
    dist=(np.min(x, axis=1))
    tp=dist[np.where(dist<20)]
    rate1=len(tp)/x.shape[0]
    rate2=len(tp)/x.shape[1]
    return rate1, rate2

def predtest(modelLarge,loss_func, opt,modeldirectory,modelfile,image,coordmatlab, directiontest,reso=1,start=0,end=None,cuda=None):
    """
    Arguments:
        modelLarge: model of the neural network used
        opt: Optimizer of the model 
        loss_func: loss function of the model
        cuda: name of the gpu which will train the model
        modeldirectory: name of the directory where the model is saved
        modelfile: name of the '.hdf5' file of the exported weights
        image: name of the 3D image (tiff) of the weaving
        coordmatlab: name of the matlab file with the coordinate of the centers of the strands
        directiontest: direction of the weaving where we will test the model, weft ou warp
	reso:coefficient of the resolution if the proportion between the image and the coordinate does not correspond
        start:if vxl starting label of the fibers of the weaving
        end: if vxl last label of the fibers of the weaving
    Returns:
        listDistFromGT: the list of the mean distance between every ground truth and its closest prediction per slice
        listDistFromDet: the list of the mean distance between every prediction and its closest ground truth per slice
        DetectionRate: the list of the sensitivity per slice
        GTfoundRate: the list of the precision per slice
        Lcentroid: the list of all the coordinate of the prediction per slice
        Lgt: the list of all the coordinate of the ground truth per slice

It also save as a .tif files the predictions as "Output.tif" and the post-process result put on the weaving as "Result.tif"
    """
#we check if we can use the gpu
    if cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]=cuda
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
    print(image)
#here we read the weaving and the coordiante of its neutral fiber if possible
    if coordmatlab is not None:
        I,M=readweaving(image, directiontest,reso,start,end,coordmatlab)
        print("M shape")
        print(type(M[0,0,0]))
        print(M.shape)
        print("number of points")
        print(M.sum())
    else:
        I=readweaving(image, directiontest,reso) 
    print("I Shape")
    print(I.shape)
#then we read the weigths of the training
    weights=''+modeldirectory+'/'+modelfile+''
    print(weights)
    if not os.path.isfile(weights):
        print("exist")
        #print(os.listdir(modeldirectory))
    modelLarge.load_weights(weights)
    modelLarge.compile(loss=loss_func, optimizer=opt)
    print(modelLarge.summary())
    Input=[]
#we resize the weaving to read it in the neural network
    for K in range (0,I.shape[0]):
        ItoPred=I[K,:,:]
        im_ori=np.expand_dims(np.insert(np.insert(ItoPred, ItoPred.shape[1],np.zeros([32-ItoPred.shape[1]%32,1]), axis=1),ItoPred.shape[0],np.zeros([32-ItoPred.shape[0]%32,1]),axis=0),axis=3)
        Input.append(im_ori)

    Input=np.array(Input)
    print("shape")
    print(Input.shape)
    print(type(Input[0,0,0,0]))
#here we produce the prediction
    Output=modelLarge.predict(Input)
    plt.figure(figsize=(16, 16))
    plt.imshow(Output[150,:,:,0])
    plt.figure(figsize=(16, 16))
    plt.imshow(Input[150,:,:,0])
    Output=Output[:,0:I.shape[1],0:I.shape[2],0]
    print(Output.shape)
    print(type(Output[K,0,0]))
    if not os.path.exists(modeldirectory+"/pred"+directiontest+image):
        os.mkdir(modeldirectory+"/pred"+directiontest+image)
        print("save prediction")
    listDistFromGT=[]
    listDistFromDet=[]
    DetectionRate=[]
    GTfoundRate=[]
    Lcentroid=[]
    Lgt=[]
    resultat=[]
    print("post processing & evaluation")
#we execute the post processing and the evaluation for every slice of the weaving
    for K in range (0,I.shape[0]):
#first we save in a folder the prediction
        plt.imsave(modeldirectory+"/pred"+directiontest+image+"/pred"+str(K)+".png",Output[K,:,:],cmap='gray')
        img=cv2.imread(modeldirectory+"/pred"+directiontest+image+"/pred"+str(K)+".png")
        img = np.array(img, dtype=np.uint8)
#here we produce the post processing
        Y,gt,centroid= PostProcessing(img,M[K,:,:])
        post=np.zeros(M[K,:,:].shape)
#here we produce with the coordinate of the prediction a numpy array to save the post processed prediction as an image
        for indx in centroid.astype(int):
            post[indx[0],indx[1]]=1
        plt.imsave(modeldirectory+"/pred"+directiontest+image+"/post"+str(K)+".png",post)
        Lcentroid.append(centroid)
        Lgt.append(gt)
#here we evaluate the prediction with the ground truth
        rate1, rate2=Evaluation(Y)
        DetectionRate.append(rate1)
        GTfoundRate.append(rate2)
        listDistFromGT.append([np.mean(np.min(Y,axis=0)),np.min(np.min(Y,axis=0)),np.max(np.min(Y,axis=0))])
        listDistFromDet.append([np.mean(np.min(Y,axis=1)),np.min(np.min(Y,axis=1)),np.max(np.min(Y,axis=1))])
        resultat.append(result(Output[K,:,:],I[K,:,:]))
    resultat=np.asarray(resultat)
    writeNpArray2tif(resultat,"resultat.tif")
    writeNpArray2tif(Output,"prediction.tif")
    return listDistFromGT,listDistFromDet,DetectionRate,GTfoundRate,Lcentroid,Lgt
