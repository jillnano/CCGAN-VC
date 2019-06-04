import os
import tensorflow as tf
from module import discriminator, generator_gatedcnn
from utils import *
from datetime import datetime

class CCGAN(object):

    def __init__(self, num_features, num_speakers, discriminator = discriminator, generator = generator_gatedcnn, mode = 'train', log_dir = './log'):

        self.num_features = num_features
        self.input_shape = [None, num_features, None] # [batch_size, num_features, num_frames]
        self.ID_vector_shape = [num_speakers]
        self.num_speakers = num_speakers

        self.discriminator = discriminator #module
        self.generator = generator #module generator_gatedcnn
        self.mode = mode

        self.build_model()
        self.optimizer_initializer()

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True

        self.saver = tf.train.Saver()
        self.sess = tf.Session(config = self.config)
        self.sess.run(tf.global_variables_initializer())
        
        if self.mode == 'train':
            self.train_step = 0
            now = datetime.now()
            self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S')) #20190101-171827
            self.writer = tf.summary.FileWriter(self.log_dir, tf.get_default_graph())
            self.generator_summaries, self.discriminator_summaries = self.summary()

    def build_model(self):

        # Placeholders for real training samples
        self.input_A_real = tf.placeholder(tf.float32, shape = self.input_shape, name = 'input_A_real')
        self.input_B_real = tf.placeholder(tf.float32, shape = self.input_shape, name = 'input_B_real')
        # Placeholders for fake generated samples
        self.input_B_fake = tf.placeholder(tf.float32, shape = self.input_shape, name = 'input_B_fake')
        # Placeholder for test samples
        self.input_A_test = tf.placeholder(tf.float32, shape = self.input_shape, name = 'input_A_test')
        
        #Placeholder for Identity Vector
        self.A_id_vector = tf.placeholder(tf.float32, shape = self.ID_vector_shape, name = 'A_id_vector')
        self.B_id_vector = tf.placeholder(tf.float32, shape = self.ID_vector_shape, name = 'B_id_vector')

        self.generation_B = self.generator(inputs = self.input_A_real, source_id = self.A_id_vector, target_id = self.B_id_vector, reuse = False, scope_name = 'generator')
        self.cycle_A = self.generator(inputs = self.generation_B, source_id = self.B_id_vector, target_id = self.A_id_vector, reuse = True, scope_name = 'generator')
        self.generation_B_identity = self.generator(inputs = self.input_B_real, source_id = self.A_id_vector, target_id = self.B_id_vector, reuse = True, scope_name = 'generator')
        self.discrimination_B_fake = self.discriminator(inputs = self.generation_B, target_id = self.B_id_vector, num_speakers = self.num_speakers, reuse = False, scope_name = 'discriminator')
       
        # Cycle loss
        self.cycle_loss = l1_loss(y = self.input_A_real, y_hat = self.cycle_A)

        # Identity loss
        self.identity_loss = l1_loss(y = self.input_A_real, y_hat = self.generation_B_identity)

        # Place holder for lambda_cycle and lambda_identity
        self.lambda_cycle = tf.placeholder(tf.float32, None, name = 'lambda_cycle')
        self.lambda_identity = tf.placeholder(tf.float32, None, name = 'lambda_identity')
        self.lambda_A2B = tf.placeholder(tf.float32, None, name = 'lambda_A2B')

        # Generator loss
        # Generator wants to fool discriminator
        self.generator_loss_A2B = ccgan_D_real_loss(y = self.B_id_vector, y_hat = self.discrimination_B_fake)

        # Merge the two generators and the cycle loss
        self.generator_loss = self.lambda_A2B * self.generator_loss_A2B + self.lambda_cycle * self.cycle_loss + self.lambda_identity * self.identity_loss

        # Discriminator loss
        self.discrimination_input_B_real = self.discriminator(inputs = self.input_B_real, target_id = self.B_id_vector, num_speakers = self.num_speakers, reuse = True, scope_name = 'discriminator')
        self.discrimination_input_B_fake = self.discriminator(inputs = self.input_B_fake, target_id = self.B_id_vector, num_speakers = self.num_speakers, reuse = True, scope_name = 'discriminator')

        self.discriminator_loss_input_B_real = ccgan_D_real_loss(y = self.B_id_vector, y_hat = self.discrimination_input_B_real)
        self.discriminator_loss_input_B_fake = l2_loss(y = tf.zeros_like(self.discrimination_input_B_fake), y_hat = self.discrimination_input_B_fake)
        self.discriminator_loss = self.discriminator_loss_input_B_real + self.discriminator_loss_input_B_fake

        # Categorize variables because we have to optimize the two sets of the variables separately
        trainable_variables = tf.trainable_variables()
        self.discriminator_vars = [var for var in trainable_variables if 'discriminator' in var.name]
        self.generator_vars = [var for var in trainable_variables if 'generator' in var.name]
        for var in trainable_variables: print(var.name)

        # Reserved for test
        self.generation_B_test = self.generator(inputs = self.input_A_test, source_id = self.A_id_vector, target_id = self.B_id_vector, reuse = True, scope_name = 'generator')
        


    def optimizer_initializer(self):

        self.generator_learning_rate = tf.placeholder(tf.float32, None, name = 'generator_learning_rate')
        self.discriminator_learning_rate = tf.placeholder(tf.float32, None, name = 'discriminator_learning_rate')
        
        self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate = self.discriminator_learning_rate, beta1 = 0.5).minimize(self.discriminator_loss, var_list = self.discriminator_vars)
        self.generator_optimizer = tf.train.AdamOptimizer(learning_rate = self.generator_learning_rate, beta1 = 0.5).minimize(self.generator_loss, var_list = self.generator_vars) 

    def train(self, input_A, input_B, lambda_cycle, lambda_identity, lambda_A2B, generator_learning_rate, discriminator_learning_rate, A_id, B_id):

        generation_B, generator_loss, _, generator_summaries = self.sess.run(
            [self.generation_B,  self.generator_loss, self.generator_optimizer, self.generator_summaries], \
            feed_dict = {self.lambda_cycle: lambda_cycle, self.lambda_identity: lambda_identity, self.lambda_A2B: lambda_A2B, self.input_A_real: input_A, self.input_B_real: input_B, self.A_id_vector: A_id, self.B_id_vector: B_id, self.generator_learning_rate: generator_learning_rate})

        self.writer.add_summary(generator_summaries, self.train_step)

        discriminator_loss, _, discriminator_summaries = self.sess.run([self.discriminator_loss, self.discriminator_optimizer, self.discriminator_summaries], \
            feed_dict = {self.input_B_real: input_B, self.discriminator_learning_rate: discriminator_learning_rate, self.input_B_fake: generation_B, self.B_id_vector: B_id})

        self.writer.add_summary(discriminator_summaries, self.train_step)

        self.train_step += 1
        
        '''
        with tf.variable_scope("discriminator/dense", reuse=True):
            w = tf.get_variable("kernel")
            print(B_id)
            print(w.eval(session=self.sess))
            
            # you can check only target end node of Discriminator backpropagates
        '''
         

        return generator_loss, discriminator_loss


    def test(self, inputs, A_id, B_id):

        generation = self.sess.run(self.generation_B_test, feed_dict = {self.input_A_test: inputs, self.A_id_vector: A_id, self.B_id_vector: B_id})

        return generation


    def save(self, directory, filename): #모델저장 ckpt

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))
        
        return os.path.join(directory, filename)

    def load(self, filepath):

        self.saver.restore(self.sess, filepath)
    
    def print_weight(self): #print min and max weight connected to id one hot vector

        with tf.variable_scope("generator/dense", reuse=True):
            w = tf.get_variable("kernel")
            print('generator_dense0', w.shape)
            print(self.sess.run(tf.reduce_max(w.eval(session=self.sess))))
            print(self.sess.run(tf.reduce_min(w.eval(session=self.sess))))
        for idx in range(1, 29):
            idx_str = str(idx)
            with tf.variable_scope("generator/dense_" + idx_str , reuse=True):
                w = tf.get_variable("kernel")
                print('generator_dense_' + idx_str,w.shape)
                print(self.sess.run(tf.reduce_max(w.eval(session=self.sess))))
                print(self.sess.run(tf.reduce_min(w.eval(session=self.sess))))
                
        with tf.variable_scope("discriminator/dense", reuse=True):
            w = tf.get_variable("kernel")
            print('discriminator_dense0', w.shape)
            print(self.sess.run(tf.reduce_max(w.eval(session=self.sess))))
            print(self.sess.run(tf.reduce_min(w.eval(session=self.sess))))
        for idx in range(1, 10):
            idx_str = str(idx)
            with tf.variable_scope("discriminator/dense_" + idx_str , reuse=True):
                w = tf.get_variable("kernel")
                print('discriminator_dense_'+ idx_str, w.shape)
                print(self.sess.run(tf.reduce_max(w.eval(session=self.sess))))
                print(self.sess.run(tf.reduce_min(w.eval(session=self.sess))))
                

        '''
        with tf.variable_scope("generator/dense_1", reuse=True):
            w = tf.get_variable("kernel")
            print(w.shape)
            print(w.eval(session=self.sess))
            print(self.sess.run(tf.reduce_max(w.eval(session=self.sess))))
        '''
        

    def summary(self):

        with tf.name_scope('generator_summaries'):
            cycle_loss_summary = tf.summary.scalar('cycle_loss', self.cycle_loss)
            identity_loss_summary = tf.summary.scalar('identity_loss', self.identity_loss)
            generator_loss_A2B_summary = tf.summary.scalar('generator_loss_A2B', self.generator_loss_A2B)
            generator_loss_summary = tf.summary.scalar('generator_loss', self.generator_loss)
            generator_summaries = tf.summary.merge([cycle_loss_summary, identity_loss_summary, generator_loss_A2B_summary, generator_loss_summary])

        with tf.name_scope('discriminator_summaries'):
            discriminator_loss_input_B_real_summary = tf.summary.scalar('discriminator_loss_input_B_real', self.discriminator_loss_input_B_real)
            discriminator_loss_input_B_fake_summary = tf.summary.scalar('discriminator_loss_input_B_fake', self.discriminator_loss_input_B_fake)
            discriminator_loss_summary = tf.summary.scalar('discriminator_loss', self.discriminator_loss)
            discriminator_summaries = tf.summary.merge([discriminator_loss_input_B_real_summary, discriminator_loss_input_B_fake_summary, discriminator_loss_summary])

        return generator_summaries, discriminator_summaries


if __name__ == '__main__':
    
    model = CCGAN(num_features = 24)
    print('Graph Compile Successeded.')