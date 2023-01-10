from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/cats/Train','input','mask',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=100,epochs=5,callbacks=[model_checkpoint])

testGene = testGenerator("data/cats/Test/input")
results = model.predict_generator(testGene,20,verbose=1)
saveResult("data/cats/Test/mask",results)


# changed pathing in testGenerator and saveResult
# added num variable
# changed epochs from 300
# reduced learning rate from 1e-4
# changed imsave method
# img_as_float
# img / 255.