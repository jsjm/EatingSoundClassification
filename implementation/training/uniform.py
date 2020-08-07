"""
version: 1.0
creator: jsjm

References: 
Urban Sound Classification using Convolutional Neural Networks with Keras: Theory and Implementation
tutorial retrieved from: https://medium.com/gradientcrescent/urban-sound-classification-using-convolutional-neural-networks-with-keras-theory-and-486e92785df4

"""

from model import model_construction

from keras_preprocessing.image import ImageDataGenerator

def classification_task(model):
        # data generation
        datagen = ImageDataGenerator(rescale=1./255., validation_split=0.25)

        train_generator = datagen.flow_from_directory(
                'mel/',
                subset="training",
                target_size=(64, 64),
                batch_size=32,
                seed=42,
                shuffle=True,
                class_mode='categorical')
        valid_generator = datagen.flow_from_directory(
                'mel/',
                subset="validation",
                target_size=(64, 64),
                batch_size=32,
                seed=42,
                shuffle=True,
                class_mode='categorical')

        #Fitting keras model
        STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
        STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
        model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=150
                        )
        res = model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID)

        # write results
        with open('res_uniform.txt', 'a') as file:
                file.write(str(res)+'\n') 

if __name__ == '__main__':
        classification_task(model_construction(20))

