import os, sys, inspect

import tensorflow as tf
import numpy as np

import source.utility as util

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def training_process(sess=None, dataset=None,
                     x=None, y_=None, training=None,
                     train_step=None, accuracy=None, loss=None, saver=None,
                     batch_size=0, steps=0):

    print("\n** Training process start!")

    te_am = dataset.test.amount
    if(batch_size > te_am):
        batch_size = te_am

    if(not(util.check_path(PACK_PATH+"/checkpoint"))):
        util.make_path(PACK_PATH+"/checkpoint")

    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []

    stepper = 1
    if(steps >= 1000):
        stepper = int(steps / 100)

    print("\n Training to "+str(steps)+" steps | Batch size: %d\n" %(batch_size))

    tra_am = dataset.train.amount
    start = 0
    end = start + batch_size
    for i in range(steps):
        if(start == 0):
            train_batch = dataset.train.next_batch(batch_size=batch_size)
        test_batch = dataset.test.next_batch(batch_size=batch_size)

        if(i % stepper == 0):
            sys.stdout.write(" Evaluation        \r")
            sys.stdout.flush()

            train_accuracy = accuracy.eval(feed_dict={x:train_batch[0], y_:train_batch[1], training:False})
            test_accuracy = accuracy.eval(feed_dict={x:test_batch[0], y_:test_batch[1], training:False})
            train_loss = loss.eval(feed_dict={x:train_batch[0], y_:train_batch[1], training:False})
            test_loss = loss.eval(feed_dict={x:test_batch[0], y_:test_batch[1], training:False})

            train_acc_list.append(train_accuracy)
            test_acc_list.append(test_accuracy)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)

            print(" step [ %d / %d ]\n Accuracy  train: %.5f  |  test: %.5f" %(i, steps, train_accuracy, test_accuracy))
            print(" CE loss   train: %.5f  |  test: %.5f" %(train_loss, test_loss))

        sys.stdout.write(" Loading next batch\r")
        sys.stdout.flush()
        train_batch = dataset.train.next_batch(batch_size=batch_size, start=start, end=end)

        sys.stdout.write(" Training          \r")
        sys.stdout.flush()
        sess.run(train_step, feed_dict={x:train_batch[0], y_:train_batch[1], training:True})

        start = end
        if(start >= tra_am):
            start = 0
        end = start + batch_size
        saver.save(sess, PACK_PATH+"/checkpoint/checker")

    test_batch = dataset.test.next_batch(batch_size=batch_size)
    test_accuracy = accuracy.eval(feed_dict={x:test_batch[0], y_:test_batch[1], training:False})
    test_loss = loss.eval(feed_dict={x:test_batch[0], y_:test_batch[1], training:False})
    print("\n Final Test accuracy, loss  | %.5f\t %.5f\n" %(test_accuracy, test_loss))

    util.save_graph_as_image(train_list=train_acc_list, test_list=test_acc_list, ylabel="accuracy")
    util.save_graph_as_image(train_list=train_loss_list, test_list=test_loss_list, ylabel="loss")

def prediction_process(sess=None, dataset=None,
                       x=None, y_=None, training=None,
                       prediction=None, saver=None,
                       validation=0):

    print("\n** Prediction process start!")

    val_am = dataset.validation.amount
    if(validation == 0):
        val_loop = val_am
    else:
        val_loop = validation
        if(val_loop > val_am):
            val_loop = val_am

    correct = 0
    if(os.path.exists(PACK_PATH+"/checkpoint/checker.index")):
        saver.restore(sess, PACK_PATH+"/checkpoint/checker")

        f = open(PACK_PATH+"/dataset/labels.txt", 'r')
        content = f.readlines()
        f.close()
        for idx in range(len(content)):
            content[idx] = content[idx][:len(content[idx])-1] # rid \n

        print("\n Prediction to "+str(val_loop)+" times")

        line_cnt = 0
        tmp_label = 0
        for i in range(val_loop):
            valid_batch = dataset.validation.next_batch(batch_size=1, nth=line_cnt)
            line_cnt += 1

            if(tmp_label != int(np.argmax(valid_batch[1]))):
                tmp_label = int(np.argmax(valid_batch[1]))

            prob = sess.run(prediction, feed_dict={x:valid_batch[0], training:False})

            print("\n Prediction")
            print(" Real:   "+str(content[int(np.argmax(valid_batch[1]))]))
            print(" Guess:  "+str(content[int(np.argmax(prob))])+"  %.2f %%" %(np.amax(prob)*100))

            if(content[int(np.argmax(valid_batch[1]))] == content[int(np.argmax(prob))]):
                correct = correct + 1

        print("\n Accuracy: %.5f" %(float(correct)/float(val_loop)))
    else:
        print("You must training first!")
