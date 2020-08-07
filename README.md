# Eating Sound Classification
Thesis log of MSc Media Technology at Leiden University


# Introduction
Food identification technology potentially benefits both food and media industries, and improves human-computer interaction.We assembled a food classification dataset based on 246 YouTube videos of 20 food types. This dataset is freely available on Kaggle. We suggest the grouped validation protocol as evaluation method to assess model performance. As a first approach, we applied Convolutional Neural Networks on this dataset. 

# Data struction
The data can be retrieved from: https://www.kaggle.com/mashijie/eating-sound-collection 
The scripts can be run on cluster computing environment and depends stronly on the file structure of data.



Here below we explain how the files are structured for our experiments:
1. Before pretreatments(when just downloaded from kaggle), the file structure looks like this:
-clips_rd
    -foodTypes
        clip_files

2. After pretreatments, these folders are added:
-mel
    -foodTypes
        -mel-spectrograms
-train
    -foodTypes
        -mel-spectrograms

-validation
    -foodTypes
        -mel-spectrograms 

3. Pairwise classification, these folders are added:
-train_binary
    -foodType1
        -mel-spectrograms
    -foodType2
        -mel-spectrograms   

-validation_binary
    -foodType1
        -mel-spectrograms
    -foodType2
        -mel-spectrograms 

# Experiments 
