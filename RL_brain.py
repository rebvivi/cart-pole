import numpy as np
import pandas as pd
import tensorflow as tf


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(   #建立 target network , evaluation network , memory
            self,
            n_actions, #要輸出多少的 action 的 qvalue
            n_features, #要接收多少個 observation 
            #用 feature 來預測 action 的值
            learning_rate=0.01,  # learning rate
            reward_decay=0.9,  # reward discount
            e_greedy=0.9,  # greedy policy
            replace_target_iter=300,  # 隔多少步把 target 的參數變成最新的參數
            memory_size=500,  # memory 容量
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0  # 紀錄運行了多少step

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        # memory 先全部設為 0  , 高度為memory_size , 長度為  (現在的observation) + (之後的observation) + reward + action

        # consist of [target_net, evaluate_net]
        self._build_net() #建立神經網路
        t_params = tf.get_collection('target_net_params') #調用所有 target network 參數 , t_params 是一個 list 
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e)
                                  for t, e in zip(t_params, e_params)] #把 evaluation 參數給 target 

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())  #activate global variables
        self.cost_his = [] #紀錄每一步的誤差

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(
            tf.float32, [None, self.n_features], name='s')  # input , 神經網路 input 的 state , 經由神經網路 output 動作的 Q 值
        self.q_target = tf.placeholder(
            tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss , q_target - q_eval = loss , 計算誤差之後 back propagation  提昇參數
            #target network 算出來的 q_target 利用 placeholder 傳入 evaluation network
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            #n_l1 : 有多少個神經元
            #c_name : 用來調用參數
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(
                    0., 0.3), tf.constant_initializer(0.1)  # config of layers , 每一層的默認參數

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'): #第一層
                w1 = tf.get_variable(
                    'w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names) #把默認的參數放入 w , b
                b1 = tf.get_variable(
                    'b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1) #相乘之後相加

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'): #第二層
                w2 = tf.get_variable(
                    'w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable(
                    'b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(
                self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(
            tf.float32, [None, self.n_features], name='s_')    # input , 輸入是下一個 state 
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable(
                    'w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable(
                    'b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable(
                    'w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable(
                    'b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_): #儲存記憶
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_)) #如果 memory 滿了 , 就覆蓋舊的數據

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1
   
   #定義 policy
    def choose_action(self, observation): #根據環境的觀測值選擇 action 的機制
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :] 
        #observation 輸入是 1 維數據 , 為了讓 tensorflow 能夠處理 , 要把維度增加 1 , 變成 2 維

        if np.random.uniform() < self.epsilon: #選最好動作
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(
                self.q_eval, feed_dict={self.s: observation}) #放入 q_evaluation 進行分析 , 輸出 action  的值
            action = np.argmax(actions_value) #選擇最大 action 的值
        else: #選隨機動作
            action = np.random.randint(0, self.n_actions) #在 action 中隨機選擇一個值
        return action

    def learn(self):  #target network 更新 , 學習 memory 中的記憶
        # check to replace target parameters
        #要不要把 evaluation 參數給 target 
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size)
        else: #抽取 memory 的 batch 數據
            sample_index = np.random.choice(
                self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params , 後 4 個
                self.s: batch_memory[:, :self.n_features],  # newest params , 前 4 個
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + \
            self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]
        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]
        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target}) #輸出誤差
        self.cost_his.append(self.cost)

        # increasing epsilon
        #從全面的 exploration  ,  到開始選擇最優的方案
        self.epsilon = self.epsilon + \
            self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self): #觀測誤差曲線
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
