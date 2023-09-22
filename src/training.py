import tensorflow as tf
import sys 
sys.path.append("..")
IMAGE_SIZE=[192,192]
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    return image


def main():
    
    file_pattern = "data\\train\\*.tfrec"  # You can use a wildcard to match multiple files
    dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(file_pattern))
    print(dataset)

    return 



if __name__=="__main__":
    main()