from keras.layers import Dense, Dropout, Activation, \
                         Flatten, Convolution2D, MaxPooling2D, \
                         BatchNormalization, Conv2D, Input,merge,AveragePooling2D,concatenate
from keras.models import Model
from utils.BilinearUpSampling import *
import keras

class SegModel(object):
        def __init__(self, input_size):
            self.input_size=input_size
            self._build_model()
            

        def relu(self,x):
            return Activation('relu')(x)

        def ResidualNet(self,nfilter,s):
            def Res_unit(x):
                BottleN = int(nfilter / 4)
                b_filter = BottleN

                x = BatchNormalization(axis=-1)(x)
                x = self.relu(x)
                ident_map = x

                x = Conv2D(b_filter,(1,1),strides=(s,s))(x)

                x = BatchNormalization(axis=-1)(x)
                x = self.relu(x)
                x = Conv2D(b_filter,3,3,border_mode='same')(x)
                x = BatchNormalization(axis=-1)(x)
                x = self.relu(x)
                x = Conv2D(nfilter,(1,1))(x)

                ident_map = Conv2D(nfilter,(1,1),strides=(s,s))(ident_map)

                out = merge([ident_map,x],mode='sum')

                return out
            return Res_unit

        def Res_Group(self,nfilter,layers,_stride):
            def Res_unit(x):
                for i in range(layers):
                    if i==0:
                        x = self.ResidualNet(nfilter,_stride)(x)
                    else:
                        x = self.ResidualNet(nfilter,1)(x)
                                   
                              
                return x
            return Res_unit

        #------------------------------Dense ASPP--------------------------------
        def PDC(self,input_layer,stride_,number_kernel,kernel_size,dconv_filters):
            l = BatchNormalization(axis=-1)(input_layer)
            l = self.relu(l)
            
            conv1 = Conv2D(number_kernel, kernel_size, activation = 'relu',padding= 'same', strides=stride_)(l)
            KRL=int(number_kernel/2)
            bl1 = Conv2D(KRL, 1, activation = 'relu', padding = 'same')(conv1)
            
            d3 = Conv2D(dconv_filters, 3, activation = 'relu', padding = 'same', dilation_rate = 3)(bl1)
            ############## D4#######
            c1 = merge([conv1,d3], mode = 'concat', concat_axis = 3)  
            KRL1=int((number_kernel+dconv_filters)/2)
            bl2 = Conv2D(KRL1, 1, activation = 'relu', padding = 'same')(c1)

            d4 = Conv2D(dconv_filters, 3, activation = 'relu', padding = 'same', dilation_rate = 6)(bl2)
            
            ########## D5###########3
            c2 = merge([conv1,d3,d4], mode = 'concat', concat_axis = 3)  
            KRL2=int(((number_kernel+dconv_filters*2))/2)
            bl3 = Conv2D(KRL2, 1, activation = 'relu', padding = 'same')(c2)

            d5 = Conv2D(dconv_filters, 3, activation = 'relu', padding = 'same', dilation_rate = 12)(bl3)
                        
            ######### 
            
            c3 = merge([conv1,d3,d4,d5], mode = 'concat', concat_axis = 3)  
            KRL3=int(((number_kernel+dconv_filters*3))/2)
            bl4 = Conv2D(KRL3, 1, activation = 'relu', padding = 'same')(c3)

            d6 = Conv2D(dconv_filters, 3, activation = 'relu', padding = 'same', dilation_rate = 18)(bl4)
               
            
            concat = merge([conv1,d3,d4,d5,d6], mode = 'concat', concat_axis = 3)  
            
            return concat
        
        def Gate(self, inp1,inp2,nf):
        
            l = BatchNormalization(axis=-1)(inp1)
            l = self.relu(l)
            conv = Conv2D(nf, 3, activation = 'relu',padding= 'same')(l)
            
            ###
            
            l2 = BatchNormalization(axis=-1)(inp2)
            l2 = self.relu(l2)
            conv1 = Conv2D(nf, 3, activation = 'relu',padding= 'same')(l2)
            inp2 = Conv2D(nf, 1, activation = 'relu',padding= 'same')(inp2)

            
            ####
            added = keras.layers.Add()([conv,conv1])
            return added
            

        
            


        def _build_model(self):
        #--------------------encoder---------    
            inp = Input(shape=(self.input_size))
            i = inp
            i = Conv2D(16,7,padding='same')(i)
        #----------------------------------------
            i = self.Res_Group(32,3,1)(i) 
            ig1=i
            #out_pdc0=self.PDC(i,8,64,1,16)
        #----------------------------------------
            i = self.Res_Group(64,3,2)(i) 
            ig2=i
            up_ig2=BilinearUpSampling2D(size = (2,2))(ig2)
            out_G1=self.Gate(ig1,up_ig2,16)
            
            

            
            
            
        #---------------------------------------
            i = self.Res_Group(128,3,2)(i) 
            ig3=i
            up_ig3=BilinearUpSampling2D(size = (2,2))(ig3)
            out_G2=self.Gate(ig2,up_ig3,32)
            
        #---------------------------------------
            i = self.Res_Group(256,3,2)(i)
            ig4=i
            up_ig4=BilinearUpSampling2D(size = (2,2))(ig4)
            out_G3=self.Gate(ig3,up_ig4,64)
            
            
        #--------------------------------------    
            i = self.Res_Group(512,3,1)(i) 
            ig5=i
            up_ig5=BilinearUpSampling2D(size = (2,2))(ig4)
            out_G4=self.Gate(ig4,ig5,64)
            

        #-----------------------decoder****************************************
            out_pdc4 = self.PDC(out_G4,1,256,1,64) # 1/8
            up_enc1=BilinearUpSampling2D(size = (2,2))(out_pdc4)
            up_enc1 = Conv2D(64, 1, activation = 'relu', padding = 'same')(up_enc1)
            
#             out_G3
            ### dec2#### 1/4
            sum1 = keras.layers.Add()([up_enc1,out_G3])
            out_pdc3 = self.PDC(sum1,1,256,1,64)
            up_enc2=BilinearUpSampling2D(size = (2,2))(out_pdc3)
            up_enc2 = Conv2D(32, 1, activation = 'relu', padding = 'same')(up_enc2)

            ###### dec3 ##### 1/2
            D0_up_enc1 = Conv2D(32, 1, activation = 'relu', padding = 'same')(up_enc1)
            D0_up_enc1=BilinearUpSampling2D(size = (2,2))(D0_up_enc1)
            
            
            
            sum2 = keras.layers.Add()([up_enc2,out_G2])
            out_pdc2 = self.PDC(sum2,1,128,1,32)

            up_enc3=BilinearUpSampling2D(size = (2,2))(out_pdc2)
            up_enc3 = Conv2D(16, 1, activation = 'relu', padding = 'same')(up_enc3)
            
            ### dec4 ### 1
            # DENSE 
            D_up_enc1 = Conv2D(16, 1, activation = 'relu', padding = 'same')(up_enc1)
            D_up_enc1=BilinearUpSampling2D(size = (4,4))(D_up_enc1)
#-----------------
            D_up_enc2=BilinearUpSampling2D(size = (2,2))(up_enc2)
            D_up_enc2 = Conv2D(16, 1, activation = 'relu', padding = 'same')(D_up_enc2)

            #
            sum3 = keras.layers.Add()([ up_enc3,out_G1])
            out_pdc1=self.PDC(sum3,1,64,1,16)
 
            i_dec = Dropout(0.5)(out_pdc1)

            conv_fo = Conv2D(1,(1, 1), activation='sigmoid', padding='same',name='A')(i_dec)
            conv_fc = Conv2D(1,(1, 1), activation='sigmoid', padding='same',name='B')(i_dec)


            model = Model(inputs=inp, outputs=[conv_fo,conv_fc])

            self.model=model
 
