import source.developed as developed
developed.print_stamp()

import argparse

import tensorflow as tf

import source.utility as util
import source.data_handler as dhand
import source.model as model
import source.sub_procedure as sproc
import source.segmentation as seg

def main():

    extensions = ["BMP", "bmp", "PNG", "png", "JPG", "jpg", "JPEG", "jpeg"]

    if((not(dhand.check())) or (FLAGS.make)):
        print("\nEnter the data path.")
        path = input(">>> ")
        dhand.make(path=path, height=250, width=150, channel=1, extensions=extensions)

    dataset = dhand.load()

    sess = tf.InteractiveSession()

    data_size, height, width, channel = dataset.train.data_size
    classes = dataset.train.class_num

    data = tf.placeholder(tf.float32, shape=[None, data_size])
    label = tf.placeholder(tf.float32, shape=[None, classes])
    training = tf.placeholder(tf.bool)

    train_step, accuracy, loss, prediction = model.convolution_neural_network(x=data, y_=label, training=training, height=height, width=width, channel=channel, classes=classes)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    print("\nDo you want to train? Y/N")
    user_need_train = input(">>> ")
    if(user_need_train.upper() == "Y"):
        sproc.training_process(sess=sess, dataset=dataset, x=data, y_=label, training=training, train_step=train_step, accuracy=accuracy, loss=loss, saver=saver, batch_size=FLAGS.batch, steps=FLAGS.steps)

    print("\nDo you want to validation? Y/N")
    user_need_valid = input(">>> ")
    if(user_need_valid.upper() == "Y"):
        sproc.prediction_process(sess=sess, dataset=dataset, x=data, y_=label, training=training, prediction=prediction, saver=saver, validation=FLAGS.validation)
    else:
        pass

    print("\nEnter the CXR image path.")
    cxr_path = input(">>> ")
    seg.extract_segments(filename=cxr_path, height=height, width=width, channel=channel, sess=sess, x_holder=data, training=training, prediction=prediction, saver=saver)

    # print("Enter the path")
    # usr_path = input(">> ")
    # if(util.check_path(usr_path)):
    #     list_dir = util.get_dirlist(path=usr_path, save=False)
    #     print(list_dir)
    #
    #     for li_d in list_dir:
    #         list_file = util.get_filelist(directory=usr_path+"/"+li_d, extensions=extensions)
    #
    #         for li_f in list_file:
    #             seg.extract_segments(filename=li_f, height=height, width=width, channel=channel, sess=sess, x_holder=data, training=training, prediction=prediction, saver=saver)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--make', type=bool, default=False, help='Default: False. Enter True to update the dataset.')
    parser.add_argument('--boost', type=int, default=0, help='Default: 0. ')
    parser.add_argument('--batch', type=int, default=10, help='Default: 10. Batches per iteration, the number of data to be training and testing.')
    parser.add_argument('--steps', type=int, default=100, help='Default: 1000')
    parser.add_argument('--validation', type=int, default=0, help='Default: 0')
    FLAGS, unparsed = parser.parse_known_args()

    main()
