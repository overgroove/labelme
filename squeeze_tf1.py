# import tensorflow as tf
import numpy as np
from PIL import Image

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

"""
Class file for SqueezeNet Model
"""

"""
Fire Module Definition
"""
def fire_module(inputs,s1x1,e1x1,e3x3,name="fire"):
    w_init = tf.truncated_normal_initializer(mean=0.0, stddev=(1.0/int(inputs.shape[2])))
    
    with tf.variable_scope(name):
        #squeeze layer
        squeeze_out = tf.layers.conv2d(inputs,filters=s1x1,kernel_size=1,strides=1,padding="VALID",kernel_initializer=w_init)
        relu_sq = tf.nn.relu(squeeze_out)
    
        #expand layer
        k1_exp = tf.layers.conv2d(relu_sq,filters=e1x1,kernel_size=1,strides=1,padding="VALID",kernel_initializer=w_init)
        k1_relu = tf.nn.relu(k1_exp)
        k3_exp = tf.layers.conv2d(relu_sq,filters=e3x3,kernel_size=3,strides=1,padding="SAME",kernel_initializer=w_init)
        k3_relu = tf.nn.relu(k3_exp)
        
        return tf.concat([k1_relu,k3_relu],axis=3)

"""
General Convolution Operation
"""
def general_conv(inputs,filters,kernel,stride=1,padding='VALID',name="conv",relu = True,weight="Xavier"):
    if str(weight) == str("Xavier"):
        w_init = tf.truncated_normal_initializer(mean=0.0,stddev=(1.0/int(inputs.shape[2])))
    else:
        w_init = tf.truncated_normal_initializer(mean=0.0,stddev=0.01)
        
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs,filters,kernel,stride,padding,kernel_initializer=w_init)
        if relu == True:
                conv = tf.nn.relu(conv)
        return conv

"""
SqueezeNet Class Definition
"""
class SqueezeNet:
    
    def __init__(self,input_shape,out_classes,lr_rate,train):
        self.lr_rate = tf.placeholder(tf.float32,name="lr_rate")
        self.out_classes = out_classes
        self.inputs = tf.placeholder(tf.float32,shape=(None,input_shape[0],input_shape[1],input_shape[2]))
        self.labels = tf.placeholder(tf.float32,shape=(None,self.out_classes))
        self.loss_v0,self.loss_v0_res,self.loss_v1 = self.model_loss(self.inputs,self.labels,train)     
        self.v0_opt,self.v0_res_opt,self.v1_opt = self.model_opti(self.loss_v0,self.loss_v0_res,self.loss_v1,self.lr_rate)
        
    #Model Definition of SqueezeNet V0
    def model_arc_v0(self,inputs,train,reuse=False):
        
        with tf.variable_scope("squeezenet_v0",reuse=reuse):
            conv1 = general_conv(inputs,filters=96,kernel=7,stride=2,padding="SAME",name="conv1",relu=True,weight="Xavier")
            pool1 = tf.layers.max_pooling2d(conv1,pool_size=3,strides=2,name="pool1")
            
            fire2 = fire_module(pool1,16,64,64,name="fire2")
            fire3 = fire_module(fire2,16,64,64,name="fire3")
            fire4 = fire_module(fire3,32,128,128,name="fire4")
        
            pool2 = tf.layers.max_pooling2d(fire4,pool_size=3,strides=2,name="pool2")
            
            fire5 = fire_module(pool2,32,128,128,name="fire5")
            fire6 = fire_module(fire5,48,192,192,name="fire6")
            fire7 = fire_module(fire6,48,192,192,name="fire7")
            fire8 = fire_module(fire7,64,256,256,name="fire8")
        
            pool3 = tf.layers.max_pooling2d(fire8,pool_size=3,strides=2,name="pool3")
        
            fire9 = fire_module(pool3,64,256,256,name="fire9")
            drop = tf.layers.dropout(fire9,rate=0.5,training=train)
        
            conv10 = general_conv(drop,filters=200,kernel=1,stride=1,padding="SAME",name="conv10",relu=True,weight="Gaussian")
        
            avg_pool = tf.layers.average_pooling2d(conv10,pool_size=13,strides=1,name="pool_end")
            pool_shape = tf.shape(avg_pool)
            logits = tf.reshape(avg_pool,shape=(pool_shape[0],pool_shape[3]))
                    
            return logits
        
    #Model Definiton of SqueezeNet V1    
    def model_arc_v1(self,inputs,train,reuse=False):
        
        with tf.variable_scope("squeezenet_v1",reuse=reuse):
            conv1 = general_conv(inputs,filters=64,kernel=3,stride=2,padding="SAME",name="conv1",relu=True,weight="Xavier")
            pool1 = tf.layers.max_pooling2d(conv1,pool_size=3,strides=2,name="pool1")
            
            fire2 = fire_module(pool1,16,64,64,name="fire2")
            fire3 = fire_module(fire2,16,64,64,name="fire3")
            
            pool2 = tf.layers.max_pooling2d(fire3,pool_size=3,strides=2,name="pool2")
            
            fire4 = fire_module(pool2,32,128,128,name="fire4")
            fire5 = fire_module(fire4,32,128,128,name="fire5")
            
            pool3 = tf.layers.max_pooling2d(fire5,pool_size=3,strides=2,name="pool3")
            
            fire6 = fire_module(pool3,48,192,192,name="fire6")
            fire7 = fire_module(fire6,48,192,192,name="fire7")
            fire8 = fire_module(fire7,64,256,256,name="fire8")       
            fire9 = fire_module(fire8,64,256,256,name="fire9")
            drop = tf.layers.dropout(fire9,rate=0.5,training=train)
        
            conv10 = general_conv(drop,filters=200,kernel=1,stride=1,padding="SAME",name="conv10",relu=True,weight="Gaussian")
        
            avg_pool = tf.layers.average_pooling2d(conv10,pool_size=13,strides=1,name="pool_end")
        
            pool_shape = tf.shape(avg_pool)
            logits = tf.reshape(avg_pool,shape=(pool_shape[0],pool_shape[3]))
        
            return logits
        
        
    #Model Definition of SqueezeNet V0 Residual     
    def model_arc_v0_res(self,inputs,train,reuse=False):
        
        with tf.variable_scope("squeezenet_v0_res",reuse=reuse):
            conv1 = general_conv(inputs,filters=96,kernel=7,stride=2,padding="SAME",name="conv1",relu=True,weight="Xavier")
            pool1 = tf.layers.max_pooling2d(conv1,pool_size=3,strides=2,name="pool1")
            
            fire2 = fire_module(pool1,16,64,64,name="fire2")
            fire3 = fire_module(fire2,16,64,64,name="fire3")
            
            bypass_23 = tf.add(fire2,fire3,name="bypass_23")
            
            fire4 = fire_module(bypass_23,32,128,128,name="fire4")
            pool2 = tf.layers.max_pooling2d(fire4,pool_size=3,strides=2,name="pool2")
            fire5 = fire_module(pool2,32,128,128,name="fire5")
            
            bypass_45 = tf.add(pool2,fire5,name="bypass_45")
            
            fire6 = fire_module(bypass_45,48,192,192,name="fire6")
            fire7 = fire_module(fire6,48,192,192,name="fire7")
            fire8 = fire_module(fire7,64,256,256,name="fire8")
        
            pool3 = tf.layers.max_pooling2d(fire8,pool_size=3,strides=2,name="pool3")
        
            fire9 = fire_module(pool3,64,256,256,name="fire9")
            
            bypass_89 = tf.add(pool3,fire9,name="bypass_89")
            
            drop = tf.layers.dropout(bypass_89,rate=0.5,training=train)
        
            conv10 = general_conv(drop,filters=200,kernel=1,stride=1,padding="SAME",name="conv10",relu=True,weight="Gaussian")
        
            avg_pool = tf.layers.average_pooling2d(conv10,pool_size=13,strides=1,name="pool_end")
        
            pool_shape = tf.shape(avg_pool)
            logits = tf.reshape(avg_pool,shape=(pool_shape[0],pool_shape[3]))
        
            return logits


    #Function to calculate loss 
    def model_loss(self,inputs,label,train):
        logits_v0 = self.model_arc_v0(inputs,train)
        logits_v0_res = self.model_arc_v0_res(inputs,train)
        logits_v1 = self.model_arc_v1(inputs,train)
        
        loss_v0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_v0,labels=label))
        loss_v0_res = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_v0_res,labels=label))
        loss_v1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_v1,labels=label))
        
        return loss_v0,loss_v0_res,loss_v1
    
    #Function to calculate prediction form models
    def model_prediction(self,inputs,train):
        logits_v0 = self.model_arc_v0(inputs,train,True)
        logits_v0_res = self.model_arc_v0_res(inputs,train,True)
        logits_v1 = self.model_arc_v1(inputs,train,True)
        
        predict_v0 = tf.nn.softmax(logits_v0)
        predict_v0_res = tf.nn.softmax(logits_v0_res)
        predict_v1 = tf.nn.softmax(logits_v1)
        
        return predict_v0,predict_v0_res,predict_v1
    
    #Function to optimize the models. 
    def model_opti(self,loss_v0,loss_v0_res,loss_v1,lr_rate):
        
        train_vars = tf.trainable_variables()
        v0_vars = [var for var in train_vars if var.name.startswith('squeezenet_v0')]
        v0_res_vars = [var for var in train_vars if var.name.startswith('squeezenet_v0_res')]
        v1_vars = [var for var in train_vars if var.name.startswith('squeezenet_v1')]        
        
        #Using Adam Optimizer 
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            v0_train_opt = tf.train.AdamOptimizer(lr_rate).minimize(loss_v0,var_list=v0_vars)
            v0_res_train_opt = tf.train.AdamOptimizer(lr_rate).minimize(loss_v0_res,var_list=v0_res_vars)
            v1_train_opt = tf.train.AdamOptimizer(lr_rate).minimize(loss_v1,var_list=v1_vars)
            
        return v0_train_opt, v0_res_train_opt, v1_train_opt

"""
Function to convert dense labels to one hot labels
"""
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

"""
Function to load image from path.
"""
def get_image_new(image_path,width,height):
    image = Image.open(image_path)
    image = image.resize([width,height],Image.BILINEAR)
    image = np.array(image,dtype=np.float32)
    image = np.subtract(image,118)
    image = np.divide(image,255)
    
    return image


"""
Function to calculate accuracy
"""	
def acc(pred,lab):
    predic = tf.equal(tf.argmax(pred,1),tf.argmax(lab,1))
    accu = tf.reduce_sum(tf.cast(predic,tf.float32))
    return accu


"""
Function to get training data path files & labels and validation data path files & labels. (Training and Cross Validation text files)
"""    
def get_data(tr_file,val_file):
    data_files = []
    with open(tr_file,"r") as f:
        data_files = f.readlines()
    tr_data = []
    tr_labels = []
    for i in data_files:
        file = i.split(' ')
        tr_data.append(file[0])
        tr_labels.append(int(file[1]))
    tr_data = np.array(tr_data)
    tr_labels = np.array(tr_labels)
    perm = np.random.permutation(tr_data.shape[0])
    tr_data = tr_data[perm]
    tr_labels = tr_labels[perm]
    data_files = []
    with open(val_file,"r") as f:
        data_files = f.readlines()
    cv_data = []
    cv_labels = []
    for i in data_files:
        file = i.split(' ')
        cv_data.append(file[0])
        cv_labels.append(int(file[1]))
    
    return tr_data,np.array(tr_labels),cv_data,np.array(cv_labels)
    

"""
Training Function 
"""
    
def train(sq_net,lr_rate,max_iter,out_classes,batch_size,tr_data_files,tr_labels,cv_data_files,cv_labels,log_file):
    train_vars = tf.trainable_variables()
    v0_vars = [var for var in train_vars if var.name.startswith('squeezenet_v0')]
    v0_res_vars = [var for var in train_vars if var.name.startswith('squeezenet_v0_res')]
    v1_vars = [var for var in train_vars if var.name.startswith('squeezenet_v1')]
    saver_v0  = tf.train.Saver(var_list=v0_vars,max_to_keep=None)
    saver_v0_res  = tf.train.Saver(var_list=v0_res_vars,max_to_keep=None)
    saver_v1  = tf.train.Saver(var_list=v1_vars,max_to_keep=None)
    print ("strating training")
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        step = 0
        bs = 0      
        max_cv_len = int(len(cv_data_files)/batch_size)*batch_size
        max_bs_len = int(len(tr_data_files)/batch_size)*batch_size
        while step < max_iter: 
            
            lr_rate = 0.0001 * (1 - step/max_iter)
            if bs >= max_bs_len:
                bs = 0 
            
            batch_files = tr_data_files[bs:(bs+batch_size)]
            batch_images = np.array([get_image_new(sample_file,227,227) for sample_file in batch_files]).astype(np.float32)
            batch_labels = np.array(dense_to_one_hot(tr_labels[bs:(bs+batch_size)],out_classes))
            
            sess.run([sq_net.v0_opt,sq_net.v0_res_opt,sq_net.v1_opt],feed_dict={sq_net.inputs:batch_images,sq_net.labels:batch_labels,sq_net.lr_rate:lr_rate})
                   
                        
            if step % 5 == 0:
                loss_v0,loss_v0_res,loss_v1 = sess.run([sq_net.loss_v0,sq_net.loss_v0_res,sq_net.loss_v1],feed_dict={sq_net.inputs:batch_images,sq_net.labels:batch_labels,sq_net.lr_rate:lr_rate})
                print ("step = %r, loss_v0 = %r loss v0_res = %r loss _v1 = %r" % (step,loss_v0,loss_v0_res,loss_v1))
            
            if step % 250 == 0: 
                cv_bs = 0
                acc_v0 = 0
                acc_v0_res = 0
                acc_v1 = 0
                                    
                while cv_bs < max_cv_len:
                        
                    cv_files = cv_data_files[cv_bs:(cv_bs+batch_size)]
                    cv_images = np.array([get_image_new(sample_file,227,227) for sample_file in cv_files]).astype(np.float32)
                    cv_img_labels = np.array(dense_to_one_hot(cv_labels[cv_bs:(cv_bs+batch_size)],out_classes))
                       
                    pred_v0,pred_v0_res,pred_v1 = sess.run(sq_net.model_prediction(sq_net.inputs,False),feed_dict={sq_net.inputs:cv_images,sq_net.labels:cv_img_labels,sq_net.lr_rate:lr_rate})
                    acc_cv_bs = sess.run(acc(pred_v0,cv_img_labels),feed_dict={sq_net.inputs:cv_images,sq_net.labels:cv_img_labels,sq_net.lr_rate:lr_rate})
                    acc_v0 = acc_v0 + acc_cv_bs
                    acc_cv_bs = sess.run(acc(pred_v0,cv_img_labels),feed_dict={sq_net.inputs:cv_images,sq_net.labels:cv_img_labels,sq_net.lr_rate:lr_rate})
                    acc_v0_res = acc_v0_res + acc_cv_bs
                    acc_cv_bs = sess.run(acc(pred_v0,cv_img_labels),feed_dict={sq_net.inputs:cv_images,sq_net.labels:cv_img_labels,sq_net.lr_rate:lr_rate})
                    acc_v1 = acc_v1 + acc_cv_bs
                    cv_bs = cv_bs + batch_size
                    print ("calc cv =%r" %(cv_bs))

                acc_v0 = (float(acc_v0)/max_cv_len) * 100.0
                acc_v0_res = (float(acc_v0_res)/max_cv_len) * 100.0
                acc_v1 = (float(acc_v1)/max_cv_len) * 100.0
                print ("Step = %r  acc_v0 = %r acc_v0_res = %r acc_v1 = %r" %(step,acc_v0,acc_v0_res,acc_v1))  
                with open(log_file,"a") as f:
                    f.write(str(str(step) + " " +  str(acc_v0) +" "+str(acc_v0_res) + " " +str(acc_v1) +"\n"))
                    
            if step % 500 == 0:
                dir_path_v0  =  model_dir + "v0_" + str(step)+"\\"
                # change second argument of restore function to the path where model weights have to be saved.
                saver_v0.save(sess,dir_path_v0,write_meta_graph=True)
                dir_path_v0_res  =  model_dir + "v0_res_" + str(step)+"\\"
                saver_v0_res.save(sess,dir_path_v0_res,write_meta_graph=True)
                dir_path_v1  =  model_dir + "v1_" + str(step)+"\\"
                saver_v1.save(sess,dir_path_v1,write_meta_graph=True)
                print ("### Model weights Saved step = %r ###" %(step))
            
            else:
                print ("step = %r" %(step))
            
            
            bs = bs + batch_size
            step = step + 1


#change model_dir to parent directory where model weights are to be saved. 
model_dir = "./model"
#change tr_file path as produced by generate_files.py file
tr_file = "squeeze\\dataset\\train_file1.txt"
#change val_file path as produced by generate_files.py file
val_file = "squeeze\\dataset\\val_file1.txt"
log_file = model_dir +"log_file.txt"
input_shape = 227,227,3
batch_size = 256
lr_rate = 0.0001
out_classes = 200
is_train = True
max_iter = 25000
tr_data_files,tr_labels,cv_data_files,cv_labels = get_data(tr_file,val_file)
tf.reset_default_graph()

sq_net = SqueezeNet(input_shape,out_classes,lr_rate,is_train)

train(sq_net,lr_rate,max_iter,out_classes,batch_size,tr_data_files,tr_labels,cv_data_files,cv_labels,log_file)
