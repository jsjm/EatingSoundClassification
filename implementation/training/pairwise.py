"""
version: 1.0
creator: jsjm

References:
Urban Sound Classification using Convolutional Neural Networks with Keras: Theory and Implementation
tutorial retrieved from: https://medium.com/gradientcrescent/urban-sound-classification-using-convolutional-neural-networks-with-keras-theory-and-486e92785df4

"""

from model import model_construction
from keras_preprocessing.image import ImageDataGenerator
import shutil, os
import argparse


def list_combinations(food_arr, combination_list):

        for i in range(0, len(food_arr)):
                for j in range(i+1, len(food_arr)):
                        combination_list.append('{0}-{1}'.format(food_arr[i], food_arr[j]))

def arrange_data(food_arr, combination_list):
        #create folders
        for i in range(0, len(combination_list)):
                os.makedirs('train_binary/{0}'.format(combination_list[i]))
                os.makedirs('validation_binary/{0}'.format(combination_list[i]))

        #copy data into combination folders
        for i in range(0, len(food_arr)):
                for j in range (i+1, len(food_arr)):
                        shutil.copytree('train/{0}'.format(food_arr[i]), 'train_binary/{0}-{1}/{2}'.format(food_arr[i], food_arr[j], food_arr[i]))
                        shutil.copytree('train/{0}'.format(food_arr[j]), 'train_binary/{0}-{1}/{2}'.format(food_arr[i], food_arr[j], food_arr[j]))
                        shutil.copytree('validation/{0}'.format(food_arr[i]), 'validation_binary/{0}-{1}/{2}'.format(food_arr[i], food_arr[j], food_arr[i]))
                        shutil.copytree('validation/{0}'.format(food_arr[j]), 'validation_binary/{0}-{1}/{2}'.format(food_arr[i], food_arr[j], food_arr[j]))

def classification_task(model, food_pair):

        # data generation
        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
                'train_binary/{0}'.format(food_pair),
                target_size=(64, 64),
                batch_size=32,
                seed=42,
                shuffle=True,
                class_mode='categorical')
        valid_generator = test_datagen.flow_from_directory(
                'validation_binary/{0}'.format(food_pair),
                target_size=(64, 64),
                batch_size=32,
                seed=42,
                shuffle=True,
                class_mode='categorical')

        #Fitting keras model
        STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
        STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
        nb_samples = len(valid_generator)
        
        model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=80
                        )
        res = model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID)

        # write results accuracy
        with open('res_matrix.txt', 'a') as file:
                file.write('{0}'.format(food_pair) + ' ' + str(res)+'\n')

        #ROC
        from sklearn.metrics import roc_curve, roc_auc_score

        preds = model.predict_generator(generator=valid_generator, steps=nb_samples)
        labels = []

        for sample in range(0,len(valid_generator)):
            # print(valid_generator[sample][1])
            for element in valid_generator[sample][1]:
                labels.append(list(element))

        roc_auc = roc_auc_score(labels, preds)

        # write results auc
        with open('res_matrix_auc.txt', 'a') as file:
                file.write('{0}'.format(food_pair) + ' ' + str(roc_auc)+'\n')



if __name__ == '__main__':

        FOODS = ['aloe', 'burger', 'cabbage', 'candied_fruits', 'carrots', 'chips', 'chocolate', 'drinks',
                'fries', 'grapes', 'gummies', 'ice_cream', 'jelly', 'noodles', 'pickles', 'pizza', 'ribs', 'salmon', 'soup', 'wings']
        combinations = []
        list_combinations(FOODS, combinations)

        # # create pair folders and arrange data (run once at start)
        # arrange_data(FOODS, combinations)

        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('combination_idx', type=int)

        args = parser.parse_args()
        print(args.combination_idx)
        with open('idx.txt', 'a') as file:
                file.write(str(args.combination_idx) +'\n')
        classification_task(model_construction(2), combinations[args.combination_idx]) # use job ID
