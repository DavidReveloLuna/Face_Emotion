# **Face_Emotion**
En este repositorio trabajaremos el entrenamiento de una red neuronal convolucional para la clasificación de emociones y la detección en tiempo real usando una cámara web. 

## Entrenamiento en google colab

El entrenamiento lo realizaremos en google colab. Vamos a configurar un entorno para trabajar con las siguientes versiones de librerías:
    
*     Tensorflow = 2.4.1
      Keras = 2.4.3 

En el siguiente enlace encuentras el código para ejecutar el entrenamiento en colab.

[FaceEmotion.ipynb](https://github.com/DavidReveloLuna/Face_Emotion/blob/master/FaceEmotion.ipynb)

## Prueba en tiempo real
<img src="https://github.com/DavidReveloLuna/Face_Emotion/blob/master/assets/ToGift.gif" width="500">

### Preparación del entorno

    $ conda create -n FaceEmotion anaconda python=3.7.7
    $ conda activate FaceEmotion
    $ pip install tensorflow==2.4.1
    $ pip install keras==2.4.3
    $ pip install numpy scipy Pillow cython matplotlib scikit-image opencv-python h5py imgaug IPython[all]
    
### Usando WebCam

Ejecutar el archivo FaceEmotionVideo.py

    $ python FaceEmotionVideo.py

# **Canal de Youtube**
[Click aquì pare ver mi canal de YOUTUBE](https://www.youtube.com/channel/UCr_dJOULDvSXMHA1PSHy2rg)
