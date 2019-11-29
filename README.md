# EIP-ASSIGNMENT_3

Assignment3
Final Validation accuracy for Base Network = 81.92


Model Definition
model = Sequential()
model.add(SeparableConv2D(48, 3, 3, border_mode='same', input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(SeparableConv2D(48, 3, 3,use_bias=False)) #rf 3, nout = 30x30x48
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(SeparableConv2D(96, 3, 3,use_bias=False)) #rf 5, nout = 28x28x96
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(MaxPooling2D(pool_size=(2, 2))) #rf 6, nout = 14x14x96

model.add(SeparableConv2D(96,3,3,use_bias=False))  #rf 10, nout = 12x12x96
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(SeparableConv2D(96,1,1,use_bias=False)) #rf 10, nout = 12x12x96
model.add(MaxPooling2D(pool_size=(2, 2))) #rf 12, nout = 6x6x96
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(SeparableConv2D(192,3,3,use_bias=False)) #rf 20 , nout = 4x4x192
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(SeparableConv2D(192,3,3,use_bias=False)) #rf 28 , nout = 2x2x192
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(MaxPooling2D(pool_size=(2, 2))) #rf 32, nout = 1x1x192
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(SeparableConv2D(10,1,1)) #rf 32, nout = 1x1x10
#model.add(SeparableConv2D(48,3,3)) 
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Activation('relu'))
#model.add(BatchNormalization())

model.add(Flatten()) #10

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.1))

# model.add(Dense(96))
# model.add(Activation('relu'))
# model.add(Dropout(0.1))

model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#SeparableConv2D




Epoch Logs

Highest val accuracy = 81.13


/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:12: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
  if sys.path[0] == '':
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:12: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras_pre..., validation_data=(array([[[..., verbose=1, steps_per_epoch=390, epochs=50)`
  if sys.path[0] == '':
Epoch 1/50
390/390 [==============================] - 24s 60ms/step - loss: 1.4636 - acc: 0.4584 - val_loss: 1.2110 - val_acc: 0.5680
Epoch 2/50
390/390 [==============================] - 22s 55ms/step - loss: 1.0355 - acc: 0.6316 - val_loss: 1.1136 - val_acc: 0.6099
Epoch 3/50
390/390 [==============================] - 21s 55ms/step - loss: 0.8908 - acc: 0.6866 - val_loss: 0.9929 - val_acc: 0.6603
Epoch 4/50
390/390 [==============================] - 21s 55ms/step - loss: 0.8039 - acc: 0.7173 - val_loss: 0.7930 - val_acc: 0.7252
Epoch 5/50
390/390 [==============================] - 21s 55ms/step - loss: 0.7452 - acc: 0.7388 - val_loss: 0.7593 - val_acc: 0.7379
Epoch 6/50
390/390 [==============================] - 21s 55ms/step - loss: 0.6999 - acc: 0.7535 - val_loss: 0.7736 - val_acc: 0.7261
Epoch 7/50
390/390 [==============================] - 21s 55ms/step - loss: 0.6766 - acc: 0.7609 - val_loss: 0.7579 - val_acc: 0.7298
Epoch 8/50
390/390 [==============================] - 21s 55ms/step - loss: 0.6382 - acc: 0.7743 - val_loss: 0.7061 - val_acc: 0.7575
Epoch 9/50
390/390 [==============================] - 21s 55ms/step - loss: 0.6089 - acc: 0.7840 - val_loss: 0.7117 - val_acc: 0.7525
Epoch 10/50
390/390 [==============================] - 21s 55ms/step - loss: 0.5966 - acc: 0.7899 - val_loss: 0.7139 - val_acc: 0.7542
Epoch 11/50
390/390 [==============================] - 21s 55ms/step - loss: 0.5739 - acc: 0.7968 - val_loss: 0.6641 - val_acc: 0.7657
Epoch 12/50
390/390 [==============================] - 21s 55ms/step - loss: 0.5553 - acc: 0.8033 - val_loss: 0.7411 - val_acc: 0.7436
Epoch 13/50
390/390 [==============================] - 21s 55ms/step - loss: 0.5379 - acc: 0.8108 - val_loss: 0.6581 - val_acc: 0.7756
Epoch 14/50
390/390 [==============================] - 22s 55ms/step - loss: 0.5210 - acc: 0.8156 - val_loss: 0.6660 - val_acc: 0.7712
Epoch 15/50
390/390 [==============================] - 21s 55ms/step - loss: 0.5121 - acc: 0.8180 - val_loss: 0.7084 - val_acc: 0.7545
Epoch 16/50
390/390 [==============================] - 21s 55ms/step - loss: 0.5019 - acc: 0.8238 - val_loss: 0.6737 - val_acc: 0.7722
Epoch 17/50
390/390 [==============================] - 21s 55ms/step - loss: 0.4860 - acc: 0.8265 - val_loss: 0.6612 - val_acc: 0.7739
Epoch 18/50
390/390 [==============================] - 21s 55ms/step - loss: 0.4837 - acc: 0.8307 - val_loss: 0.6675 - val_acc: 0.7717
Epoch 19/50
390/390 [==============================] - 21s 55ms/step - loss: 0.4639 - acc: 0.8365 - val_loss: 0.6808 - val_acc: 0.7660
Epoch 20/50
390/390 [==============================] - 21s 55ms/step - loss: 0.4590 - acc: 0.8386 - val_loss: 0.7164 - val_acc: 0.7504
Epoch 21/50
390/390 [==============================] - 21s 55ms/step - loss: 0.4481 - acc: 0.8402 - val_loss: 0.6605 - val_acc: 0.7755
Epoch 22/50
390/390 [==============================] - 21s 55ms/step - loss: 0.4463 - acc: 0.8418 - val_loss: 0.5738 - val_acc: 0.8046
Epoch 23/50
390/390 [==============================] - 21s 55ms/step - loss: 0.4371 - acc: 0.8444 - val_loss: 0.6037 - val_acc: 0.7923
Epoch 24/50
390/390 [==============================] - 21s 55ms/step - loss: 0.4246 - acc: 0.8494 - val_loss: 0.7181 - val_acc: 0.7538
Epoch 25/50
390/390 [==============================] - 21s 55ms/step - loss: 0.4189 - acc: 0.8525 - val_loss: 0.6616 - val_acc: 0.7763
Epoch 26/50
390/390 [==============================] - 21s 55ms/step - loss: 0.4136 - acc: 0.8534 - val_loss: 0.6625 - val_acc: 0.7747
Epoch 27/50
390/390 [==============================] - 21s 55ms/step - loss: 0.4020 - acc: 0.8563 - val_loss: 0.6404 - val_acc: 0.7879
Epoch 28/50
390/390 [==============================] - 21s 55ms/step - loss: 0.4033 - acc: 0.8566 - val_loss: 0.6907 - val_acc: 0.7641
Epoch 29/50
390/390 [==============================] - 21s 55ms/step - loss: 0.3934 - acc: 0.8598 - val_loss: 0.6381 - val_acc: 0.7849
Epoch 30/50
390/390 [==============================] - 21s 55ms/step - loss: 0.3865 - acc: 0.8605 - val_loss: 0.6645 - val_acc: 0.7840
Epoch 31/50
390/390 [==============================] - 21s 55ms/step - loss: 0.3788 - acc: 0.8637 - val_loss: 0.6316 - val_acc: 0.7883
Epoch 32/50
390/390 [==============================] - 21s 55ms/step - loss: 0.3783 - acc: 0.8643 - val_loss: 0.6966 - val_acc: 0.7720
Epoch 33/50
390/390 [==============================] - 21s 55ms/step - loss: 0.3690 - acc: 0.8674 - val_loss: 0.5807 - val_acc: 0.8046
Epoch 34/50
390/390 [==============================] - 21s 55ms/step - loss: 0.3583 - acc: 0.8723 - val_loss: 0.5874 - val_acc: 0.8070
Epoch 35/50
390/390 [==============================] - 21s 55ms/step - loss: 0.3591 - acc: 0.8725 - val_loss: 0.6630 - val_acc: 0.7801
Epoch 36/50
390/390 [==============================] - 21s 55ms/step - loss: 0.3604 - acc: 0.8713 - val_loss: 0.6790 - val_acc: 0.7829
Epoch 37/50
390/390 [==============================] - 21s 55ms/step - loss: 0.3498 - acc: 0.8741 - val_loss: 0.5856 - val_acc: 0.8034
Epoch 38/50
390/390 [==============================] - 21s 54ms/step - loss: 0.3519 - acc: 0.8744 - val_loss: 0.5999 - val_acc: 0.8031
Epoch 39/50
390/390 [==============================] - 21s 55ms/step - loss: 0.3523 - acc: 0.8742 - val_loss: 0.6691 - val_acc: 0.7806
Epoch 40/50
390/390 [==============================] - 22s 55ms/step - loss: 0.3379 - acc: 0.8769 - val_loss: 0.5901 - val_acc: 0.8060
Epoch 41/50
390/390 [==============================] - 21s 55ms/step - loss: 0.3373 - acc: 0.8780 - val_loss: 0.5957 - val_acc: 0.8049
Epoch 42/50
390/390 [==============================] - 21s 55ms/step - loss: 0.3362 - acc: 0.8781 - val_loss: 0.6958 - val_acc: 0.7784
Epoch 43/50
390/390 [==============================] - 21s 55ms/step - loss: 0.3329 - acc: 0.8809 - val_loss: 0.6821 - val_acc: 0.7793
Epoch 44/50
390/390 [==============================] - 21s 55ms/step - loss: 0.3292 - acc: 0.8803 - val_loss: 0.6611 - val_acc: 0.7899
Epoch 45/50
390/390 [==============================] - 21s 55ms/step - loss: 0.3238 - acc: 0.8827 - val_loss: 0.7199 - val_acc: 0.7653
Epoch 46/50
390/390 [==============================] - 21s 55ms/step - loss: 0.3181 - acc: 0.8854 - val_loss: 0.6196 - val_acc: 0.8006
Epoch 47/50
390/390 [==============================] - 21s 55ms/step - loss: 0.3180 - acc: 0.8870 - val_loss: 0.7745 - val_acc: 0.7576
Epoch 48/50
390/390 [==============================] - 21s 55ms/step - loss: 0.3129 - acc: 0.8882 - val_loss: 0.6008 - val_acc: 0.8035
Epoch 49/50
390/390 [==============================] - 21s 55ms/step - loss: 0.3067 - acc: 0.8894 - val_loss: 0.6408 - val_acc: 0.7945
Epoch 50/50
390/390 [==============================] - 21s 55ms/step - loss: 0.3133 - acc: 0.8885 - val_loss: 0.5861 - val_acc: 0.8113
Model took 1072.39 seconds to train

Accuracy on test data is: 81.13
