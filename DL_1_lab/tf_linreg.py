import tensorflow as tf

## 1. definicija računskog grafa
# podatci i parametri
X  = tf.placeholder(tf.float32, [None])
Y_ = tf.placeholder(tf.float32, [None])
a = tf.Variable(0.0)
b = tf.Variable(0.0)

# afini regresijski model
Y = a * X + b

# kvadratni gubitak
loss = (Y-Y_)**2

# optimizacijski postupak: gradijentni spust
trainer = tf.train.GradientDescentOptimizer(0.1)
train_op = trainer.minimize(loss)
tf_grads = tf.gradients(loss, [a, b])

# operations together perform gradient descent minimize function
grads = trainer.compute_gradients(loss, var_list=[a, b]) # this is the first operation of minimizing
apply_placeholder_op = trainer.apply_gradients(grads)  # this is the second operation of minimizing

printout = tf.Print(grads, [a, b], message="Gradients:")

## 2. inicijalizacija parametara
sess = tf.Session()
sess.run(tf.initialize_all_variables())

## 3. učenje
# neka igre počnu!
for i in range(100):
    # if we put in X and Y_ 3 coords, it breaks, returning Nans and Infs
    # works for [0, 1, 2] / [1, 3, 5]  but not for [ 1, 2, 3] / [3, 5, 7]
    val_loss, val_a, val_b, var_grad, apply = sess.run([loss, a, b, grads, apply_placeholder_op],
                                                          feed_dict={X: [1, 2], Y_: [3, 5]})
    print(i, val_loss, val_a, val_b, "\t", var_grad[0])

    # val_loss, val_a, val_b, _, grads = sess.run([loss, a, b, train_op, tf_grads],
    #                                             feed_dict={X: [1, 2], Y_: [3, 5]})
    # print(i, val_loss, val_a, val_b, "\t", grads)

    # TODO: figure out tf.Print
    # val_loss, val_a, val_b, _, grads = sess.run([loss, a, b, apply_placeholder_op, printout],
    #                                              feed_dict={X: [1, 2], Y_: [3, 5]})
    # print(i, val_loss, val_a, val_b,"\t")
