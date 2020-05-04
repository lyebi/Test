from __future__ import print_function
from collections import namedtuple
import numpy as np
import tensorflow as tf
from model import LSTMPolicy, MetaPolicy,MetaPolicy2,MetaPolicy3,MetaPolicy4,curiosity_network
import six.moves.queue as queue
import scipy.signal
import threading
import distutils.version
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')
import cv2

def cosine_similarity(u, v, eps=1e-8):
    u_norm = tf.sqrt(tf.reduce_sum(tf.square(u),axis=1))
    v_norm = tf.sqrt(tf.reduce_sum(tf.square(v),axis=1))
    uv = tf.reduce_sum(tf.multiply(u,v),axis=1)
    cosin = uv/(u_norm * v_norm)
    return cosin

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class A3C(object):
    def __init__(self, env, task, visualise, test=False):
        """
An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
should be computed.
"""

        self.env = env
        self.task = task

        self.meta_action_size1 = 32
        self.meta_action_size2=37
        self.meta_action_size3=5
        self.meta_action_size4=30
        self.meta_action_size = self.meta_action_size1+self.meta_action_size2 + self.meta_action_size3+ self.meta_action_size4

        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        if test:
           worker_device = "/job:eval/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = LSTMPolicy(env.observation_space.shape, env.action_space.n, self.meta_action_size)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)
                self.meta_network1 = MetaPolicy(env.observation_space.shape, self.meta_action_size1)
                self.meta_network2=  MetaPolicy2(env.observation_space.shape, self.meta_action_size2)
                self.meta_network3=  MetaPolicy3(env.observation_space.shape, self.meta_action_size3)
                self.meta_network4=  MetaPolicy4(env.observation_space.shape, self.meta_action_size4)
                self.meta_curiosity_network = curiosity_network(env.observation_space.shape,18)


        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = LSTMPolicy(env.observation_space.shape, env.action_space.n, self.meta_action_size)
                self.local_meta_network1 = meta_pi1= MetaPolicy(env.observation_space.shape, self.meta_action_size1)
                self.local_meta_network2 = meta_pi2= MetaPolicy2(env.observation_space.shape, self.meta_action_size2)
                self.local_meta_network3 = meta_pi3= MetaPolicy3(env.observation_space.shape, self.meta_action_size3)
                self.local_meta_network4 = meta_pi4= MetaPolicy4(env.observation_space.shape, self.meta_action_size4)
                self.local_curiosity_network = local_curiosity =  curiosity_network(env.observation_space.shape, 18)

                pi.global_step = self.global_step

            self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")

            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf = tf.nn.softmax(pi.logits)


            # the "policy gradients" loss:  its derivative is precisely the policy gradient
            # notice that self.ac is a placeholder that is provided externally.
            # adv will contain the advantages, as calculated in process_rollout
            pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

            # loss of value function
            vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))
            entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

            bs = tf.to_float(tf.shape(pi.x)[0])
            self.loss = pi_loss + 0.5 * vf_loss - entropy * 0.01


            self.visualise = visualise

            grads = tf.gradients(self.loss, pi.var_list)

            actor_summary = [
                tf.summary.scalar("model/policy_loss", pi_loss / bs),
                tf.summary.scalar("model/value_loss", vf_loss / bs),
                tf.summary.scalar("model/entropy", entropy / bs),
                tf.summary.image("model/state", pi.x),
                tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads)),
                tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
                ]

            self.summary_op = tf.summary.merge(actor_summary)

            grads, _ = tf.clip_by_global_norm(grads, 40.0)

            # This is sync ops which copy weights from shared space to the local.
            self.sync = tf.group(
                *(
                    [ v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)]
                 ))


            grads_and_vars = list(zip(grads, self.network.var_list))
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])
            # each worker has a different set of adam optimizer parameters
            opt = tf.train.AdamOptimizer(1e-4)
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
            self.summary_writer = None
            self.local_steps = 0

            ###################################
            ########## META CONTROLLER1 ########
            ###################################
            self.meta_ac1 = tf.placeholder(tf.float32, [None, self.meta_action_size1], name="meta_ac1")
            self.meta_adv1 = tf.placeholder(tf.float32, [None], name="meta_adv1")
            self.meta_r1 = tf.placeholder(tf.float32, [None], name="meta_r1")

            meta_log_prob_tf1 = tf.nn.log_softmax(meta_pi1.logits)
            meta_prob_tf1 = tf.nn.softmax(meta_pi1.logits)

            meta_pi_loss1 = - tf.reduce_sum(tf.reduce_sum(meta_log_prob_tf1 * self.meta_ac1, [1]) * self.meta_adv1)
            meta_vf_loss1 = 0.5 * tf.reduce_sum(tf.square(meta_pi1.vf - self.meta_r1))

            # entropy
            meta_entropy1 = - tf.reduce_sum(meta_prob_tf1 * meta_log_prob_tf1)
            meta_bs1 = tf.to_float(tf.shape(meta_pi1.x)[0])

            self.meta_loss1 = meta_pi_loss1 + 0.5 * meta_vf_loss1 - meta_entropy1 * 0.01
            meta_grads1 = tf.gradients(self.meta_loss1, meta_pi1.var_list)
            meta_grads1, _ = tf.clip_by_global_norm(meta_grads1, 40.0)

            self.meta_sync1 = tf.group(
                *(
                    [ v1.assign(v2) for v1, v2 in zip(meta_pi1.var_list, self.meta_network1.var_list)]
                 ))

            meta_grads_and_vars1 = list(zip(meta_grads1, self.meta_network1.var_list))
            meta_opt1 = tf.train.AdamOptimizer(1e-4)
            self.meta_train_op1 = meta_opt1.apply_gradients(meta_grads_and_vars1)

            # meta_summary1 = [
            #     tf.summary.scalar("meta_model/policy_loss", meta_pi_loss1 / meta_bs1),
            #     tf.summary.scalar("meta_model/value_loss", meta_vf_loss1 / meta_bs1),
            #     tf.summary.scalar("meta_model/entropy", meta_entropy1 / meta_bs1),
            #     tf.summary.scalar("meta_model/grad_global_norm", tf.global_norm(meta_grads1)),
            #     tf.summary.scalar("meta_model/var_global_norm", tf.global_norm(meta_pi1.var_list))
            # ]
            # self.meta_summary_op = tf.summary.merge(meta_summary1)
            self.beta1 = 0.75
            # self.beta1 = 0.60


            ###################################
            ########## META CONTROLLER:pixel_control ########
            ###################################
            self.meta_ac2 = tf.placeholder(tf.float32, [None, self.meta_action_size2], name="meta_ac2")
            self.meta_adv2 = tf.placeholder(tf.float32, [None], name="meta_adv2")
            self.meta_r2 = tf.placeholder(tf.float32, [None], name="meta_r2")

            meta_log_prob_tf2 = tf.nn.log_softmax(meta_pi2.logits)
            meta_prob_tf2 = tf.nn.softmax(meta_pi2.logits)

            meta_pi_loss2 = - tf.reduce_sum(tf.reduce_sum(meta_log_prob_tf2 * self.meta_ac2, [1]) * self.meta_adv2)
            meta_vf_loss2 = 0.5 * tf.reduce_sum(tf.square(meta_pi2.vf - self.meta_r2))

            # entropy
            meta_entropy2 = - tf.reduce_sum(meta_prob_tf2 * meta_log_prob_tf2)
            self.meta_loss2 = meta_pi_loss2 + 0.5 * meta_vf_loss2 - meta_entropy2 * 0.01
            meta_grads2 = tf.gradients(self.meta_loss2, meta_pi2.var_list)
            meta_grads2, _ = tf.clip_by_global_norm(meta_grads2, 40.0)

            self.meta_sync2 = tf.group(
                *(
                    [v1.assign(v2) for v1, v2 in zip(meta_pi2.var_list, self.meta_network2.var_list)]
                ))

            meta_grads_and_vars2 = list(zip(meta_grads2, self.meta_network2.var_list))
            meta_opt2 = tf.train.AdamOptimizer(1e-4)
            self.meta_train_op2 = meta_opt2.apply_gradients(meta_grads_and_vars2)
            meta_summary2 = [
                tf.summary.scalar("meta_model/policy_loss", meta_pi_loss1 / meta_bs1),
                tf.summary.scalar("meta_model/value_loss", meta_vf_loss1 / meta_bs1),
                tf.summary.scalar("meta_model/entropy", meta_entropy1 / meta_bs1),
                tf.summary.scalar("meta_model/grad_global_norm", tf.global_norm(meta_grads1)),
                tf.summary.scalar("meta_model/var_global_norm", tf.global_norm(meta_pi1.var_list)),

                tf.summary.scalar("meta_model/policy_loss2", meta_pi_loss2 / meta_bs1),
                tf.summary.scalar("meta_model/value_loss2", meta_vf_loss2 / meta_bs1),
                tf.summary.scalar("meta_model/entropy2", meta_entropy2 / meta_bs1),
                tf.summary.scalar("meta_model/grad_global_norm2", tf.global_norm(meta_grads2)),
                tf.summary.scalar("meta_model/var_global_norm2", tf.global_norm(meta_pi2.var_list))
            ]
            self.meta_summary_op = tf.summary.merge(meta_summary2)

            self.beta2 = 0.75

            #########################################################################################################
            ########## META CONTROLLER1 : direction instruction
            #######################################################################################################
            self.meta_ac3 = tf.placeholder(tf.float32, [None, self.meta_action_size3], name="meta_ac3")
            self.meta_adv3 = tf.placeholder(tf.float32, [None], name="meta_adv3")
            self.meta_r3 = tf.placeholder(tf.float32, [None], name="meta_r3")

            meta_log_prob_tf3 = tf.nn.log_softmax(meta_pi3.logits)
            meta_prob_tf3 = tf.nn.softmax(meta_pi3.logits)

            meta_pi_loss3 = - tf.reduce_sum(tf.reduce_sum(meta_log_prob_tf3 * self.meta_ac3, [1]) * self.meta_adv3)
            meta_vf_loss3 = 0.5 * tf.reduce_sum(tf.square(meta_pi3.vf - self.meta_r3))

            # entropy
            meta_entropy3 = - tf.reduce_sum(meta_prob_tf3 * meta_log_prob_tf3)
            meta_bs3= tf.to_float(tf.shape(meta_pi3.x)[0])

            self.meta_loss3 = meta_pi_loss3 + 0.5 * meta_vf_loss3 - meta_entropy3 * 0.01
            meta_grads3 = tf.gradients(self.meta_loss3, meta_pi3.var_list)
            meta_grads3, _ = tf.clip_by_global_norm(meta_grads3, 40.0)

            self.meta_sync3 = tf.group(
                *(
                    [v1.assign(v2) for v1, v2 in zip(meta_pi3.var_list, self.meta_network3.var_list)]
                ))

            meta_grads_and_vars3 = list(zip(meta_grads3, self.meta_network3.var_list))
            meta_opt3 = tf.train.AdamOptimizer(1e-4)
            self.meta_train_op3 = meta_opt3.apply_gradients(meta_grads_and_vars3)
            self.beta3 = 0.75

            #########################################################################################################
            ########## META CONTROLLER1 : offset goal
            #######################################################################################################
            self.meta_ac4 = tf.placeholder(tf.float32, [None, self.meta_action_size4], name="meta_ac4")
            # self.meta4_diff = tf.placeholder(tf.float32,[None,self.meta_action_size4],name="meta_diff")
            self.meta_adv4 = tf.placeholder(tf.float32, [None], name="meta_adv4")
            self.meta_r4 = tf.placeholder(tf.float32, [None], name="meta_r4")

            meta_log_prob_tf4 = tf.nn.log_softmax(meta_pi4.logits)
            meta_prob_tf4 = tf.nn.softmax(meta_pi4.logits)

            meta_pi_loss4 = - tf.reduce_sum(tf.reduce_sum(meta_log_prob_tf4 * self.meta_ac4, [1]) * self.meta_adv4)
            meta_vf_loss4 = 0.5 * tf.reduce_sum(tf.square(meta_pi4.vf - self.meta_r4))

            # entropy
            meta_entropy4 = - tf.reduce_sum(meta_prob_tf4 * meta_log_prob_tf4)
            # meta_bs4 = tf.to_float(tf.shape(meta_pi4.x)[0])

            self.meta_loss4 = meta_pi_loss4 + 0.5 * meta_vf_loss4
            meta_grads4 = tf.gradients(self.meta_loss4, meta_pi4.var_list)
            meta_grads4, _ = tf.clip_by_global_norm(meta_grads4, 40.0)

            self.meta_sync4 = tf.group(
                *(
                    [v1.assign(v2) for v1, v2 in zip(meta_pi4.var_list, self.meta_network4.var_list)]
                ))

            meta_grads_and_vars4 = list(zip(meta_grads4, self.meta_network4.var_list))
            meta_opt4 = tf.train.AdamOptimizer(1e-4)
            self.meta_train_op4 = meta_opt4.apply_gradients(meta_grads_and_vars4)
            self.beta4 = 0.75

            ##########################################################################################
            ####################





    def start(self, sess, summary_writer):
        self.summary_writer = summary_writer

        # Initialise Actor
        # Initialise last_state and last_features
        self.last_state = self.env.reset()
        self.last_result=self.env.reset()[6:10,20:60]
        self.last_features = self.local_network.get_initial_features()
        self.last_action = np.zeros(self.env.action_space.n)
        self.last_reward = [0]
        self.length = 0
        self.rewards = 0
        self.ex_rewards = 0
        self.in_rewards = 0
        self.in_rewards2 = 0
        self.in_rewards3 = 0
        self.in_rewards4 = 0



        # Initialise Meta controller
        self.last_meta_state1 = self.env.reset()
        self.last_meta_state2 = self.env.reset()
        self.last_meta_state3 = self.env.reset()
        self.last_meta_state4 = self.env.reset()
        self.last_meta_features1 = self.local_meta_network1.get_initial_features()
        self.last_meta_features2 = self.local_meta_network2.get_initial_features()
        self.last_meta_features3 = self.local_meta_network2.get_initial_features()
        self.last_meta_features4 = self.local_meta_network4.get_initial_features()
        self.last_meta_action1 = np.zeros(self.meta_action_size1)
        self.last_meta_action2 = np.zeros(self.meta_action_size2)
        self.last_meta_action3 = np.zeros(self.meta_action_size3)
        self.last_meta_action4 = np.zeros(self.meta_action_size4)
        self.last_meta_reward1 = [0]
        self.last_meta_reward2 = [0]
        self.last_meta_reward3 = [0]
        self.last_meta_reward4 = [0]

        #
        self.last_conv_feature = np.zeros(self.meta_action_size1)
        self.last_conv_feature2 = np.zeros(self.meta_action_size4)

    def process(self, sess):
        """
        Everytime process is called.
        The meta_network get sync.
        The actor_process is run for 20 times.
        The meta_network calculate gradient and update
        """
        sess.run(self.meta_sync1)
        sess.run(self.meta_sync2)
        sess.run(self.meta_sync3)
        sess.run(self.meta_sync4)


        terminal_end = False
        # TODO: tune this too
        num_local_steps = 20
        env = self.env
        policy1 = self.local_meta_network1
        policy2 = self.local_meta_network2
        policy3 = self.local_meta_network3
        policy4 = self.local_meta_network4

        states1  = []
        actions1 = []
        rewards1 = []
        values1  = []
        r1       = 0.0
        terminal1= False
        features1= []
        prev_actions1 = []
        prev_rewards1 = []

        states2  = []
        actions2 = []
        rewards2 = []
        values2  = []
        r2       = 0.0
        terminal2= False
        features2= []
        prev_actions2 = []
        prev_rewards2 = []

        states3 = []
        actions3 = []
        rewards3 = []
        values3 = []
        r3 = 0.0
        terminal3 = False
        features3 = []
        prev_actions3 = []
        prev_rewards3 = []

        states4 = []
        actions4 = []
        rewards4 = []
        diffs=[]
        values4 = []
        r4 = 0.0
        terminal4 = False
        features4 = []
        prev_actions4 = []
        prev_rewards4 = []



        for _local_step in range(num_local_steps):


            fetched1 = policy1.act(self.last_meta_state1, self.last_meta_features1[0],
                                 self.last_meta_features1[1], self.last_meta_action1,
                                 self.last_meta_reward1)

            fetched2 = policy2.act(self.last_meta_state2, self.last_meta_features2[0],
                                 self.last_meta_features2[1], self.last_meta_action2,
                                 self.last_meta_reward2)

            fetched3 = policy3.act(self.last_meta_state3, self.last_meta_features3[0],
                                   self.last_meta_features3[1], self.last_meta_action3,
                                   self.last_meta_reward3)
            fetched4 = policy4.act(self.last_meta_state4, self.last_meta_features4[0],
                                   self.last_meta_features4[1], self.last_meta_action4,
                                   self.last_meta_reward4)


            action1, value1_, features1_ = fetched1[0], fetched1[1], fetched1[2:]
            action2, value2_, features2_ = fetched2[0], fetched2[1], fetched2[2:]
            action3, value3_, features3_ = fetched3[0], fetched3[1], fetched3[2:]
            action4, value4_, features4_ = fetched4[0], fetched4[1], fetched4[2:]

            state1, reward1,reward2,reward3,reward4, terminal1, info1= self.actor_process(sess, [action1,action2,action3,action4])
            # collect experience
            states1 += [self.last_meta_state1]
            states2 += [self.last_meta_state2]
            states3 += [self.last_meta_state3]
            states4 += [self.last_meta_state4]

            actions2+=[action2]
            actions1 += [action1]
            actions3 += [action3]
            actions4 += [action4]


            rewards1 += [reward1]
            rewards2 += [reward2]
            rewards3 += [reward3]
            rewards4 += [reward4]


            values1 += [value1_]
            values2 += [value2_]
            values3 += [value3_]
            values4 += [value4_]

            features1 += [self.last_meta_features1]
            features2 += [self.last_meta_features2]
            features3 += [self.last_meta_features3]
            features4 += [self.last_meta_features4]


            prev_actions1 += [self.last_meta_action1]
            prev_actions2 += [self.last_meta_action2]
            prev_actions3 += [self.last_meta_action3]
            prev_actions4 += [self.last_meta_action4]

            prev_rewards1 += [self.last_meta_reward1]
            prev_rewards2 += [self.last_meta_reward2]
            prev_rewards3 += [self.last_meta_reward3]
            prev_rewards4 += [self.last_meta_reward4]

            # update state
            self.last_meta_state1 = state1
            self.last_meta_state2 = state1
            self.last_meta_state3 = state1
            self.last_meta_state4 = state1

            self.last_meta_features1 = features1_
            self.last_meta_features2 = features2_
            self.last_meta_features3 = features3_
            self.last_meta_features4 = features4_

            self.last_meta_action1 = action1
            self.last_meta_action2 = action2
            self.last_meta_action3 = action3
            self.last_meta_action4 = action4

            self.last_meta_reward1 = [reward1]
            self.last_meta_reward2 = [reward2]
            self.last_meta_reward3 = [reward3]
            self.last_meta_reward4 = [reward4]

            if terminal1:
                self.last_meta_features1= policy1.get_initial_features()
                self.last_meta_features2= policy2.get_initial_features()
                self.last_meta_features3= policy3.get_initial_features()
                self.last_meta_features4= policy4.get_initial_features()
                break
        if not terminal1:
            r1 = policy1.value(self.last_meta_state1, self.last_meta_features1[0],
                                 self.last_meta_features1[1], self.last_meta_action1,
                                 self.last_meta_reward1)
            r2 = policy2.value(self.last_meta_state2, self.last_meta_features2[0],
                                 self.last_meta_features2[1], self.last_meta_action2,
                                 self.last_meta_reward2)
            r3 = policy3.value(self.last_meta_state3, self.last_meta_features3[0],
                               self.last_meta_features3[1], self.last_meta_action3,
                               self.last_meta_reward3)
            r4 = policy4.value(self.last_meta_state4, self.last_meta_features4[0],
                               self.last_meta_features4[1], self.last_meta_action4,
                               self.last_meta_reward4)


        # Process rollout
        gamma = 0.99
        lambda_ = 1.0

        batch_si = np.asarray(states1)

        batch_a = np.asarray(actions1)
        batch_a2 = np.asarray(actions2)
        batch_a3 = np.asarray(actions3)
        batch_a4 = np.asarray(actions4)


        rewards_plus_v = np.asarray(rewards1 + [r1])
        rewards_plus_v2 = np.asarray(rewards2 + [r2])
        rewards_plus_v3 = np.asarray(rewards3 + [r3])
        rewards_plus_v4 = np.asarray(rewards4 + [r4])


        rewards = np.asarray(rewards1)
        rewards2 = np.asarray(rewards2)
        rewards3 = np.asarray(rewards3)
        rewards4 = np.asarray(rewards4)

        vpred_t = np.asarray(values1 + [r1])
        # vpred_t2 = np.asarray(values2 + [r1])
        vpred_t2 = np.asarray(values2 + [r2])
        vpred_t3 = np.asarray(values3 + [r3])
        vpred_t4 = np.asarray(values4 + [r4])


        batch_r = discount(rewards_plus_v, gamma)[:-1]
        batch_r2 = discount(rewards_plus_v2, gamma)[:-1]
        batch_r3 = discount(rewards_plus_v3, gamma)[:-1]
        batch_r4 = discount(rewards_plus_v4, gamma)[:-1]

        delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
        delta_t2 = rewards2 + gamma * vpred_t2[1:] - vpred_t2[:-1]
        delta_t3 = rewards3 + gamma * vpred_t3[1:] - vpred_t3[:-1]
        delta_t4 = rewards4 + gamma * vpred_t4[1:] - vpred_t4[:-1]
        # this formula for the advantage comes "Generalized Advantage Estimation":
        # https://arxiv.org/abs/1506.02438
        batch_adv = discount(delta_t, gamma * lambda_)
        batch_adv2 = discount(delta_t2, gamma * lambda_)
        batch_adv3 = discount(delta_t3, gamma * lambda_)
        batch_adv4 = discount(delta_t4, gamma * lambda_)

        batch_prev_a = np.asarray(prev_actions1)
        batch_prev_a2 = np.asarray(prev_actions2)
        batch_prev_a3 = np.asarray(prev_actions3)
        batch_prev_a4 = np.asarray(prev_actions4)

        batch_prev_r = np.asarray(prev_rewards1)
        batch_prev_r2 = np.asarray(prev_rewards2)
        batch_prev_r3 = np.asarray(prev_rewards3)
        batch_prev_r4 = np.asarray(prev_rewards4)

        features = features1[0]
        features2 = features2[0]
        features3 = features3[0]
        features4 = features4[0]

        # Gradient Calculation
        fetches = [self.meta_summary_op, self.meta_train_op1,self.meta_train_op2, self.meta_train_op3,self.meta_train_op4,self.global_step]

        feed_dict = {
            self.local_meta_network1.x: batch_si,
            self.meta_ac1: batch_a,
            self.meta_adv1: batch_adv,
            self.meta_r1: batch_r,
            self.local_meta_network1.state_in[0]: features[0],
            self.local_meta_network1.state_in[1]: features[1],
            self.local_meta_network1.prev_action: batch_prev_a,
            self.local_meta_network1.prev_reward: batch_prev_r,

            self.local_meta_network2.x: batch_si,
            self.meta_ac2: batch_a2,
            self.meta_adv2: batch_adv2,
            self.meta_r2: batch_r2,
            self.local_meta_network2.state_in[0]: features2[0],
            self.local_meta_network2.state_in[1]: features2[1],
            self.local_meta_network2.prev_action: batch_prev_a2,
            self.local_meta_network2.prev_reward: batch_prev_r2,

            self.local_meta_network3.x: batch_si,
            self.meta_ac3: batch_a3,
            self.meta_adv3: batch_adv3,
            self.meta_r3: batch_r3,
            self.local_meta_network3.state_in[0]: features3[0],
            self.local_meta_network3.state_in[1]: features3[1],
            self.local_meta_network3.prev_action: batch_prev_a3,
            self.local_meta_network3.prev_reward: batch_prev_r3,

            self.local_meta_network4.x: batch_si,
            self.meta_ac4:batch_a4,
            self.meta_adv4: batch_adv4,
            self.meta_r4: batch_r4,
            self.local_meta_network4.state_in[0]: features4[0],
            self.local_meta_network4.state_in[1]: features4[1],
            self.local_meta_network4.prev_action: batch_prev_a4,
            self.local_meta_network4.prev_reward: batch_prev_r4
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)
        if self.task == 0:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()


    def actor_process(self, sess, meta_action):
        """
        Every time actor_process is called.e
        The network get sync.
        The environment is run for 20 steps or until termination.
        The worker calculates gradients and then one update to the shared weight is made.
        (one local step = one update  =< 20 env steps )
        (global step is the number of frames)
        """
        sess.run(self.sync)  # copy weights from shared to local

        # Environment run for 20 steps or less
        terminal_end = False
        num_local_steps = 100
        env = self.env
        policy = self.local_network
        curiosity_policy = self.local_curiosity_network


        states  = []
        actions = []
        rewards = []
        values  = []
        r       = 0.0
        terminal= False
        features= []
        prev_actions = []
        prev_rewards = []
        extrinsic_rewards = []
        extrinsic_rewards1 = []
        extrinsic_rewards2 = []
        extrinsic_rewards3 = []
        extrinsic_rewards4 = []
        intrinsic_rewards1=[]
        intrinsic_rewards2=[]
        intrinsic_rewards3=[]
        intrinsic_rewards4=[]


        # select patch 1 in 36. each patch is 14x14
        # idx = 6*x + y where x:[0,5], y[0:5], idx:[0,35]
        # x =  idx // 6
        idx = meta_action[0].argmax()
        idx2 = meta_action[1].argmax()
        idx3 = meta_action[2].argmax()
        idx4 = meta_action[3].argmax()

        print(np.shape(meta_action[1]))
        print(np.shape(meta_action[2]))
        print(np.shape(meta_action[3]))
        meta_action_total=np.array(list(meta_action[0])+list(meta_action[1])+list(meta_action[2])+list(meta_action[3]))


        pos_x = idx2 // 6
        pos_y = idx2 - 6*pos_x
        goal_patch = np.zeros([84, 84, 3])
        if idx2 != 37:
           goal_patch[ 14 * pos_x: 14 * (pos_x + 1) + 1, 14*pos_y: 14*(pos_y+1) +1 ] = 1

        weights = policy.get_weights(self.last_state,meta_action_total)[0][0]
        weights = np.exp(weights)/sum(np.exp(weights))



        for _local_step in range(num_local_steps):
            # Take a step
            fetched = policy.act(self.last_state, self.last_features[0], self.last_features[1],
                                 self.last_action, self.last_reward, meta_action_total)

            action, value_, features_ = fetched[0], fetched[1], fetched[2:]

            curiosity_features = curiosity_policy.act(self.last_state,self.last_action)


            # argmax to convert from one-hot
            state, reward, terminal, info = env.step(action.argmax())

            result=state[6:10,20:60]
            if np.sum(result)!=np.sum(self.last_result):
                flags=1
            else:
                flags=0


            # clip reward
            # reward = min(1, max(-1, reward))
            if reward>0:
                reward/=100.0
                # self.beta1=self.beta1+0.1/20000000
            else:
                reward = min(1, max(-1, reward))

            # Intrinsic reward
            # Pixel control
            pixel_changes = (state - self.last_state)**2
            # mean square error normalized by all pixel_changes
            intrinsic_reward2 = 0.05 * np.sum(pixel_changes * goal_patch ) / np.sum( pixel_changes + 1e-5)


            if action.argmax()  in [0,2] and idx3==0:
                intrinsic_reward3=0.01
            elif action.argmax() in [1,10] and idx3==1 :
                intrinsic_reward3 = 0.01
            elif action.argmax() in [4,7,9,12,15,17] and idx3==2:
                intrinsic_reward3=0.01
            elif action.argmax() in [3,6,8,11,14,16] and idx3==3:
                intrinsic_reward3 = 0.01
            elif action.argmax() in [5,13] and idx3==4:
                intrinsic_reward3=0.01
            else:
                intrinsic_reward3=0



            # Feature control [selectivity (Bengio et al., 2017)]
            conv_feature = policy.get_conv_feature(state)[0][0]
            sel = np.abs(conv_feature[idx] - self.last_conv_feature[idx])
            sel = sel / ( np.sum( np.abs(conv_feature - self.last_conv_feature)) + 1e-5)
            self.last_conv_feature = conv_feature
            intrinsic_reward = 0.05 * sel


            conv_feature2 = policy.get_encoding(state)[0][0]
            sel2 = np.abs(conv_feature2[idx4] - self.last_conv_feature2[idx4])
            sel2 = sel2 / (np.sum(np.abs(conv_feature2 - self.last_conv_feature2)) + 1e-5)
            self.last_conv_feature2 = conv_feature2
            intrinsic_reward4 = 0.05 * sel2


            real_next_features = policy.get_feature(state)[0][0]
            curiosity_diff = np.sqrt(np.sum(np.square(real_next_features-curiosity_features)))
            intrinsic_curiosity_reward = 0.05 * curiosity_diff




            # conv_feature2[conv_feature2>0]=1
            # self.last_conv_feature2[self.last_conv_feature2>0]=1
            # relu_unit_change=np.sum(conv_feature2)-np.sum(self.last_conv_feature2)
            # aux_reward=relu_unit_change/3872
            # intrinsic_reward+=aux_reward
            # self.last_conv_feature2 = conv_feature2


            # print(intrinsic_reward2)
            # print('intrinstic reward:', intrinsic_reward)
            # print('intrinstic reward2:', intrinsic_reward2)

            intrinsic_rewards1+=[intrinsic_reward]
            intrinsic_rewards2+=[intrinsic_reward2]
            intrinsic_rewards3+=[intrinsic_reward3]
            intrinsic_rewards4+=[intrinsic_reward4]


            if intrinsic_reward>0:
                reward1=reward
            else:
                reward1=0

            if intrinsic_reward2>0:
                reward2=reward
            else:
                reward2=0

            if intrinsic_reward3>0:
                reward3=reward
            else:
                reward3=0

            if intrinsic_reward4>0:
                reward4=reward
            else:
                reward4=0

            # record extrinsic reward
            extrinsic_rewards += [reward]
            extrinsic_rewards1 += [reward1]
            extrinsic_rewards2 += [reward2]
            extrinsic_rewards3 += [reward3]
            extrinsic_rewards4 += [reward4]
            self.ex_rewards += reward
            self.in_rewards += intrinsic_reward
            self.in_rewards2+=intrinsic_reward2
            self.in_rewards3+=intrinsic_reward3
            self.in_rewards4+=intrinsic_reward4

            # Apply intrinsic reward
            beta = self.beta1
            reward = beta * reward + (1.0 - beta) *(intrinsic_reward*weights[0]+intrinsic_reward2*weights[1]+
                                                    intrinsic_reward3*weights[2]+intrinsic_reward4*weights[3] +
                                                    intrinsic_curiosity_reward)

            if self.visualise:
                vis = state - 0.5 * state * goal_patch + 0.5 * goal_patch
                vis = cv2.resize(vis, (400,400))
                cv2.imshow('img', vis)
                cv2.waitKey(5)

            # collect the experience
            states += [self.last_state]
            actions += [action]
            rewards += [reward]
            values += [value_]
            features += [self.last_features]
            prev_actions += [self.last_action]
            prev_rewards += [self.last_reward]


            self.length += 1
            self.rewards += reward

            self.last_state = state
            # self.last_result= result
            self.last_features = features_
            self.last_action = action
            self.last_reward = [reward]

            if info:
                summary = tf.Summary()
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))


                self.summary_writer.add_summary(summary, policy.global_step.eval())
                self.summary_writer.flush()

            timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            # timestep_limit = 5000
            if terminal or self.length >= timestep_limit:
                terminal_end = True
                if self.length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    self.last_state = env.reset()
                self.last_features = policy.get_initial_features()
                print("Episode finished. Sum of rewards: %d. Length: %d" % (self.rewards, self.length))

                summary = tf.Summary()
                summary.value.add(tag='global/episode_shaped_reward', simple_value=self.rewards)
                summary.value.add(tag='global/shaped_reward_per_time', simple_value=self.rewards/self.length)
                summary.value.add(tag='global/episode_extrinsic_reward', simple_value=self.ex_rewards)
                summary.value.add(tag='global/episode_intrinsic_reward', simple_value=self.in_rewards)
                summary.value.add(tag='global/episode_intrinsic_reward2', simple_value=self.in_rewards2)
                summary.value.add(tag='global/episode_intrinsic_reward3', simple_value=self.in_rewards3)
                summary.value.add(tag='global/episode_intrinsic_reward4', simple_value=self.in_rewards4)

                self.summary_writer.add_summary(summary, policy.global_step.eval())
                self.summary_writer.flush()

                self.length = 0
                self.rewards = 0
                self.ex_rewards = 0
                self.in_rewards = 0
                self.in_rewards2=0
                self.in_rewards3=0
                self.in_rewards4=0

                break

        if not terminal_end:
            r = policy.value(self.last_state, self.last_features[0],
                             self.last_features[1], self.last_action,
                             self.last_reward, meta_action_total)

        # Process rollout
        gamma = 0.99
        lambda_ = 1.0
        batch_si = np.asarray(states)
        batch_a = np.asarray(actions)
        rewards_plus_v = np.asarray(rewards + [r])
        rewards = np.asarray(rewards)
        vpred_t = np.asarray(values + [r])
        batch_r = discount(rewards_plus_v, gamma)[:-1]
        delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
        # this formula for the advantage comes "Generalized Advantage Estimation":
        # https://arxiv.org/abs/1506.02438
        batch_adv = discount(delta_t, gamma * lambda_)
        batch_prev_a = np.asarray(prev_actions)
        batch_prev_r = np.asarray(prev_rewards)
        # print('shape features:',np.shape(features))

        features = features[0] # only use first feature into dynamic rnn
        # print('shape features[0]',np.shape(features))


        # Batch meta action
        batch_meta_ac = np.repeat([meta_action_total], len(batch_si), axis=0)

        # Gradient Calculation
        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0
        if should_compute_summary:
            fetches = [self.summary_op, self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]

        feed_dict = {
            self.local_network.x: batch_si,
            self.ac: batch_a,
            self.adv: batch_adv,
            self.r: batch_r,
            self.local_network.state_in[0]: features[0],
            self.local_network.state_in[1]: features[1],
            self.local_network.prev_action: batch_prev_a,
            self.local_network.prev_reward: batch_prev_r,
            self.local_network.meta_action: batch_meta_ac
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()
        self.local_steps += 1

        # discount extrinsic reward for the meta controller
        gamma = 0.99
        # early rewards are better?
        discount_filter = np.array([gamma**i for i in range(len(extrinsic_rewards))])
        extrinsic_rewards= np.sum(discount_filter * extrinsic_rewards)

        return self.last_state, np.sum(extrinsic_rewards), np.sum(extrinsic_rewards),np.sum(extrinsic_rewards),np.sum(extrinsic_rewards),terminal_end, None

    def evaluate(self,sess):

        global_step = sess.run(self.global_step)
        sess.run(self.meta_sync1)
        sess.run(self.meta_sync2)
        sess.run(self.sync)

        meta_policy = self.local_meta_network1
        policy = self.local_network
        env = self.env
        rewards_stat = []
        length_stat = []
        # average over 100 episode?
        for episode in range(100):
            terminal = False

            last_state = env.reset()
            last_meta_state = last_state
            last_features = policy.get_initial_features()
            last_meta_features = meta_policy.get_initial_features()
            last_meta_action = np.zeros(self.meta_action_size)
            last_meta_reward = [0]
            last_action = np.zeros(self.env.action_space.n)
            last_reward = [0]
            rewards = 0
            length = 0
            last_conv_feature = np.zeros(self.meta_action_size)


            while not terminal:

                fetched = meta_policy.act(last_meta_state, last_meta_features[0],
                                          last_meta_features[1], last_meta_action, last_meta_reward)
                meta_action, meta_value_, meta_features_ = fetched[0], fetched[1], fetched[2:]
                meta_reward = 0
                idx = meta_action.argmax()
                for _ in range(20*5):
                    fetched = policy.act(last_state, last_features[0], last_features[1],
                                     last_action, last_reward, meta_action)
                    action, value_, features_ = fetched[0], fetched[1], fetched[2:]
                    state, reward, terminal, info = env.step(action.argmax())

                    if self.visualise:
                        vis = cv2.resize(state , (500,500))
                        cv2.imshow('img', vis)
                        cv2.waitKey(10)

                    env_reward = reward

                    # clip reward
                    reward = min(1, max(-1, reward))

                    # Feature control [selectivity (Bengio et al., 2017)]
                    conv_feature = policy.get_conv_feature(state)[0][0]
                    sel = np.abs(conv_feature[idx] - last_conv_feature[idx])
                    sel = sel / ( np.sum( np.abs(conv_feature - last_conv_feature) ) + 1e-5)
                    last_conv_feature = conv_feature

                    intrinsic_reward = 0.05 * sel

                    # Apply intrinsic reward
                    beta = self.beta
                    shaped_reward = beta * reward + (1.0 - beta) * intrinsic_reward


                    length += 1
                    rewards += env_reward
                    last_state = state
                    last_features = features_
                    last_action = action
                    last_reward = [shaped_reward]
                    meta_reward += reward

                    timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
                    if terminal or length >= timestep_limit:
                        terminal = True
                        break

                last_meta_state = last_state
                last_meta_features = meta_features_
                last_meta_action = meta_action
                last_meta_reward = [meta_reward]

                if terminal:
                    break

            rewards_stat.append(rewards)
            length_stat.append(length)

        summary = tf.Summary()
        summary.value.add(tag='Eval/Average_Reward', simple_value=np.mean(rewards_stat))
        summary.value.add(tag='Eval/SD_Reward', simple_value=np.std(rewards_stat))
        summary.value.add(tag='Eval/Average_Lenght', simple_value=np.mean(length_stat))
        self.summary_writer.add_summary(summary, global_step)
        self.summary_writer.flush()
