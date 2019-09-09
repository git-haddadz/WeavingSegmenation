from skimage import io
import numpy as np
import os
import tensorflow as tf
from skimage.morphology import dilation, square
import scipy.io
import keras
import matplotlib.pyplot as plt
import h5py
def dataset(I,M, nb_samples, start, end, SEpar,pas,dim_patchx,dim_patchy,both=False,I2=None,M2=None,start2=None,end2=None):
    """
    Arguments:
        I: input image from which the input of the dataset is generated
        M: numpy representating the center of the ground truth from which the dataset is generated
	both:Toke
        nb_samples: number of patchs wwe have to generate
        start: starting slice from which we create the dataset
        end: last slice from which we create the dataset
        SEpar: height of the square
    Returns:
        x: set of input patchs
        y: set of ground truth patchs
    """
    x=[]
    y=[]
    if (both==True):
        nb_samples=int(nb_samples/2)
    for i in range (0,nb_samples):
        klayer=np.random.randint(int(start/pas),int(end/pas))*pas
        if (M.shape[1]>dim_patchx):
            xini=np.random.randint(M.shape[1]-dim_patchx)
        else:
            xini=0
        yini=np.random.randint(M.shape[2]-dim_patchy)
        if (M.shape[1]>dim_patchx):
            x.append(I[klayer,xini:(xini+dim_patchx),yini:(yini+dim_patchy)])
            y.append((dilation(M[klayer,xini:(xini+dim_patchx),yini:(yini+dim_patchy)]>0,square(SEpar))))
        else:
            x.append(np.insert(I[klayer,xini:(xini+dim_patchx),yini:(yini+dim_patchx)], M.shape[1],np.zeros([dim_patchx-M.shape[1],dim_patchx]), axis=0))
            y.append(np.insert((dilation(M[klayer,xini:(xini+dim_patchx),yini:(yini+dim_patchy)]>0,square(SEpar))), M.shape[1],np.zeros([dim_patchx-M.shape[1],dim_patchx]), axis=0))
        if (both==True):
            klayer=np.random.randint(start2,end2)
            if (M2.shape[1]>dim_patchx):
                xini=np.random.randint(M2.shape[1]-dim_patchx)
            else:
                xini=0
            yini=np.random.randint(M2.shape[2]-dim_patchy)
            if (M2.shape[1]>dim_patchx):
                x.append(I2[klayer,xini:(xini+dim_patchx),yini:(yini+dim_patchy)])
                y.append((dilation(M2[klayer,xini:(xini+dim_patchx),yini:(yini+dim_patchy)]>0,square(SEpar))))
            else:
                x.append(np.insert(I2[klayer,xini:(xini+dim_patchx),yini:(yini+dim_patchy)], M.shape[1],np.zeros([dim_patchx-M2.shape[1],dim_patchx]), axis=0))
                y.append(np.insert((dilation(M2[klayer,xini:(xini+dim_patchx),yini:(yini+dim_patchy)]>0,square(SEpar))), M.shape[1],np.zeros([dim_patchx-M2.shape[1],dim_patchx]), axis=0))
    x=np.expand_dims(np.asarray(x),axis=3)
    y=np.expand_dims(np.asarray(y),axis=3)
    index = np.random.randint(nb_samples)
    print(index)
    plt.figure(figsize=(16,16))
    plt.subplot(1, 2, 1)
    plt.imshow(x[index, :, :, 0])
    plt.title("Input")
    plt.subplot(1, 2, 2)
    plt.imshow(y[index, :, :, 0])
    plt.title("Ground truth")
    plt.figure(figsize=(16,16))
    plt.subplot(1, 2, 1)
    plt.imshow(x[index+1, :, :, 0])
    plt.title("Input")
    plt.subplot(1, 2, 2)
    plt.imshow(y[index+1, :, :, 0])
    plt.title("Ground truth")
    return x,y
	
def readweaving(image, direction,reso=1,startfiber=0,endfiber=None,coordmatlab=None):
    """
    Arguments:
        image: name of the 3D image (tiff) of the weaving
        direction: direction of the weaving you want to read weft ou warp
        OPTION
	reso:coefficient of the resolution if the proportion between the image and the coordinate does not correspond
        coordmatlab:If we have it name of the matlab file with the coordinate of the centers of the strands
        startfiber:if vxl starting label of the fibers of the weaving
        endfiber: if vxl last label of the fibers of the weaving

    Return: 
        I: Numpy of the part of weaving in the direction wanted
        IF coordmatlab NOT Empty
        M: Numpy of associated ground truth
        """
    name=""+image+".tif"
    I = io.imread(name)
    print("Image shape")
    print(I.shape)
    if coordmatlab is not None:
        if (coordmatlab[len(coordmatlab)-3:len(coordmatlab)]=="vxl"):
                infile = open(coordmatlab)
                Copy = False
                M=np.zeros(I.shape,dtype=np.uint8)
                Labeli=0
                if direction=="warp":
                    print("warp")
                    for line in infile:
                        ls1=line.split()
                        if (line.strip()[0:9] == "WARP_Yarn") or (line.strip()[0:9] == "WARP_YARN"):
                            Copy=True
                            Labeli+=1
                            print("id = ",Labeli)
                        elif line.strip() == "---":
                            Copy=False
                        elif Copy:
                            ls=line.split()
                            if len(ls)> 2:                                            
                                if (Labeli>start)and(Labeli<end):
                                    M[int(round(float(ls[1])/reso)),int(round(float(ls[3])/reso)),int(round(float(ls[2])/reso))]=Labeli
                elif direction=="weft":
                    print("weft")
                    for line in infile:
                        ls1=line.split()
                        if (line.strip()[0:9] == "WEFT_Yarn") or (line.strip()[0:9] == "WEFT_YARN"):
                            Copy=True
                            Labeli+=1
                            print(Labeli)
                        elif line.strip() == "---":
                            Copy=False
                        elif Copy:
                            ls=line.split()
                            if len(ls)> 2:
                                #print("id = ",Labeli,": ",int(round(float(ls[1])/reso)),int(round(float(ls[3])/reso)),int(round(float(ls[2])/reso)))
                                M[int(round(float(ls[1])/reso)),int(round(float(ls[3])/reso)),int(round(float(ls[2])/reso))]=Labeli
                                
                    I=I.T
                    M=M.T
                else:
                    raise NameError("no direction")
        elif(coordmatlab[len(coordmatlab)-3:len(coordmatlab)]=="mat"):
                mat = scipy.io.loadmat(coordmatlab)
                print('Keys')
                print(mat.keys())
                M=np.zeros(I.shape,dtype=np.uint8)
                Labeli=1
                if direction=="warp":
                    print("warp")
                    for i in range(len(mat['RESULTATS_Warp'][0])):
                        for indx in mat['RESULTATS_Warp'][0][i][0].astype(int):
                            M[indx[0],indx[2],indx[1]]=Labeli
                        Labeli+=1
                elif direction=="weft":
                    print("weft")
                    Labeli=1
                    for i in range(len(mat['RESULTATS_Weft'][0])):
                        for indx in mat['RESULTATS_Weft'][0][i][0].astype(int):
                            M[indx[0],indx[2],indx[1]]=Labeli
                        Labeli+=1
                    I=I.T
                    M=M.T
                else:
                    raise NameError("no direction")
        else:
            print("mauvais format")
        print(M.shape)
        print("number of points")
        print(M.sum())
        start=0
        print(M.shape)
        end=M.shape[0]
        while(M[start,:,:].sum()==0):
                start=start+1
        print("start ",start)
        while(M[end-1,:,:].sum()==0):
                end=end-1
        print("end ",end)
        print(M[start:end,:,:].sum())
        return I[start:end,:,:],M[start:end,:,:]
    else:
        return I
                
def traincenter(model, loss_func, opt, cuda,image,coordmatlab, directiontrain,nb_train_samples,nb_val_samples,batch_size,dim_patch_x,dim_patch_y,size_neutral_fiber,pas,nb_epoch,both=False,image2=None,coord2=None):
    """
    Arguments:
        model: model of the neural network used
        opt: Optimizer of the model 
        loss_func: loss function of the model
        cuda: name of the gpu which will train the model
        image: name of the 3D image (tiff) of the weaving
        coordmatlab: name of the matlab file with the coordinate of the centers of the strands
        directiontrain: direction of the weaving where we will train the model, weft ou warp
        nb_train_samples: number of patchs of training set
        nb_val_samplesnumber of patchs of validation set
        batch_size: batch size during the training
        nb_epoch: number of epoch during the training
    
    It will save the model by exporting the weights in a '.hdf5' file, in the directory created
    
    Returns:
        val_loss: validation loss during the training
        loss: loss during the training
        modeldirectory: name of the directory where the model is saved
        modelfile: name of the '.hdf5' file of the exported weights
    
    
    """
    os.environ["CUDA_VISIBLE_DEVICES"]=cuda
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if (both==True):
        modelname="Train"+image+image2+directiontrain+"Direction"
    else:
        modelname="Train"+image+directiontrain+"Direction"
    I,M=readweaving(image, directiontrain,coordmatlab=coordmatlab)
    print("M shape")
    print(M.shape)
    print("I Shape")
    print(I.shape)
    print("number of points")
    print(M.sum())
    print("training set")
    if (both==True):
        I2,M2=readweaving(image2, directiontrain,coord2)
        X_train,Y_train=dataset(I,M, nb_train_samples,0,int(M.shape[0]*(2/3)),size_neutral_fiber,pas,dim_patch_x,dim_patch_y,True,I2,M2,0,int(M2.shape[0]*(2/3)))
    else:
        X_train,Y_train=dataset(I,M, nb_train_samples,0,int(M.shape[0]*(2/3)),size_neutral_fiber,pas,dim_patch_x,dim_patch_y)
    print("shape")
    print(X_train.shape)
    print(Y_train.shape)
    print("validation set")
    if (both==True):
        X_val,Y_val=dataset(I,M, nb_val_samples,int(M.shape[0]*(2/3))+1,M.shape[0],size_neutral_fiber,pas,dim_patch_x,dim_patch_y,True,I2,M2,int(M2.shape[0]*(2/3))+1,M2.shape[0])
    else:
        X_val,Y_val=dataset(I,M, nb_val_samples,int(M.shape[0]*(2/3))+1,M.shape[0],size_neutral_fiber,pas,dim_patch_x,dim_patch_y)
    print("shape")
    print(X_val.shape)
    print(Y_val.shape)
    model.compile(loss=loss_func, optimizer=opt)
    print(model.summary())
    earlystop=keras.callbacks.EarlyStopping(monitor='val_loss',
                            patience=3, verbose=1,
                            mode='auto', baseline=None,
                            restore_best_weights=True)
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        epochs=nb_epoch,
                        validation_data=(X_val, Y_val),
                        shuffle=True,
                        verbose=1,
                        callbacks=[earlystop])
    
    print("end of the training")
    val_loss=history.history['val_loss']
    loss=history.history['loss']
    print("Best validation loss: %.5f" % (np.min(history.history['val_loss'])))
    print("at: %d" % np.argmin(history.history['val_loss']))
    modeldirectory="directoryDLon"+modelname
    if not os.path.exists(modeldirectory):
        os.mkdir(modeldirectory)
    modelfile='weight'+modelname+'.hdf5'
    model.save_weights(modeldirectory+"/"+modelfile)
    return val_loss, loss, modeldirectory, modelfile

def transferlearning(model,loss_func, opt,modeldirectory,modelfile,cuda,image,coordmatlab, directiontest):
    os.environ["CUDA_VISIBLE_DEVICES"]=cuda
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    modelname="Transfer"+image+directiontrain+"Direction"
    I,M=readweaving(image, directiontrain,coordmatlab)
    print("M shape")
    print(M.shape)
    print("I Shape")
    print(I.shape)
    print("number of points")
    print(M.sum())
    print("training set")
    X_train,Y_train=dataset(I,M, nb_train_samples,0,int(M.shape[0]*(2/3)), 15)
    print("shape")
    print(X_train.shape)
    print(Y_train.shape)
    print("validation set")
    X_val,Y_val=dataset(I,M, nb_val_samples,int(M.shape[0]*(2/3))+1,M.shape[0], 15)
    print("shape")
    print(X_val.shape)
    print(Y_val.shape)

    weights=''+modeldirectory+'/'+modelfile+''
    modelLarge.load_weights(weights)
    model.compile(loss=loss_func, optimizer=opt)
    print(model.summary())
    earlystop=keras.callbacks.EarlyStopping(monitor='val_loss',
                            patience=3, verbose=1,
                            mode='auto', baseline=None,
                            restore_best_weights=True)
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        epochs=nb_epoch,
                        validation_data=(X_val, Y_val),
                        shuffle=True,
                        verbose=1,
                        callbacks=[earlystop])
    print("end of the transfer")
    val_loss=history.history['val_loss']
    loss=history.history['loss']
    print("Best validation loss: %.5f" % (np.min(history.history['val_loss'])))
    print("at: %d" % np.argmin(history.history['val_loss']))
    modeldirectory="directoryDLon"+modelname
    if not os.path.exists(modeldirectory):
        os.mkdir(modeldirectory)
    modelfile='weight'+modelname+'.hdf5'
    model.save_weights(modeldirectory+"/"+modelfile)
    return val_loss, loss, modeldirectory, modelfile
