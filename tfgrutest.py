import tensorflow as tf
import tensorflow.contrib as tf_ct
import numpy as np
import maxout_util
from tensorflow.python import debug as tfdbg

import trajectory_data
import util


LR = 0.01
MAXOUT_K = 5
HIDDEN_NODES = 50
FEATURE_INTERVALS = 20
EMBEDDING_DIMS = 50
BATCH_SIZE = 32
SEQ_LEN = 25
MODE = 4
SEQ_LEN = [BATCH_SIZE,HIDDEN_NODES,EMBEDDING_DIMS]
ITER_NUM = 30000
LOG_DIR = "E:/PythonProjects/RNNTest/LOG"
V_MAX = 0
V_MIN = 0
V_SD_MAX = 0
V_SD_MIN = 0
V_AVG_MAX = 0
V_AVG_MIN = 0


def normalization(train_data):

    #计算embed_v 即每个点的速度。
    embed_b_v = np.zeros([ITER_NUM*BATCH_SIZE*HIDDEN_NODES,1],dtype=np.float32)
    step = HIDDEN_NODES +1
    for i in range(ITER_NUM):
        for j in range(BATCH_SIZE):
            for k in range(HIDDEN_NODES):
                start = (i*BATCH_SIZE+j) * step + k
                dis = util.jwd2dis(train_data[start][0],
                                   train_data[start][1],
                                   train_data[start + 1][0],
                                   train_data[start + 1][1])
                embed_b_v[(i*BATCH_SIZE+j) * HIDDEN_NODES + k][0] = dis / ((train_data[start + 1][3] - train_data[start][3]) * 3600 * 24)
    #算出所有V的max 和min
    v_max = np.argmax(embed_b_v)
    v_min = np.argmin(embed_b_v)

    #计算embed_v_avg 即每段的平均速度
    embed_b_v_avg = np.zeros([ITER_NUM*BATCH_SIZE*HIDDEN_NODES,1],dtype=np.float32)
    for i in range(ITER_NUM):
        for j in range(BATCH_SIZE):
            start = (i*BATCH_SIZE+j)*HIDDEN_NODES
            end = start + HIDDEN_NODES
            embed_b_v_avg[start:end] = np.mean(embed_b_v[start:end])

    v_avg_max = embed_b_v_avg.argmax()
    v_avg_min = embed_b_v_avg.argmin()

    #计算embed_v_sd 即每段的标准差
    embed_b_v_sd = np.zeros([ITER_NUM*BATCH_SIZE*HIDDEN_NODES,1],dtype=np.float32)
    for i in range(ITER_NUM):
        for j in range(BATCH_SIZE):
            start = (i*BATCH_SIZE+j)*HIDDEN_NODES
            end = start + HIDDEN_NODES
            embed_b_v_sd[start:end] = np.std(embed_b_v[start:end])

    v_sd_max = embed_b_v_sd.argmax()
    v_sd_min = embed_b_v_sd.argmin()

    #归一化

    for i in range(embed_b_v.shape[0]):
        embed_b_v[i] = (embed_b_v[i]-v_min)/(v_max-v_min)
        embed_b_v_avg[i] = (embed_b_v_avg[i]-v_avg_min)/(v_avg_max-v_avg_min)
        embed_b_v_sd[i] = (embed_b_v_sd[i]-v_sd_min)/(v_sd_max-v_sd_min)


    return  embed_b_v,embed_b_v_avg,embed_b_v_sd



def to_onehot(v,avg,vsd):
    embed_v = np.floor(v*20)
    embed_v = np.array(embed_v, dtype=np.int32)
    embed_v = np.array(trajectory_data.dense_to_one_hot(embed_v, 20), dtype=np.float32)


    embed_v_avg = np.floor(avg*20)
    embed_v_avg = np.array(embed_v_avg, dtype=np.int32)
    embed_v_avg = np.array(trajectory_data.dense_to_one_hot(embed_v_avg, 20), dtype=np.float32)


    embed_v_sd = np.floor(vsd*20)
    embed_v_sd = np.array(embed_v_sd,dtype=np.int32)
    embed_v_sd = np.array(trajectory_data.dense_to_one_hot(embed_v_sd, 20), dtype=np.float32)

    return embed_v,embed_v_avg,embed_v_sd



#maxout 激励层
def maxout_activator(inputs):

    return maxout_util.maxout(inputs, MAXOUT_K, HIDDEN_NODES, name="maxout_activation")


def embedding_before(train_input,batch_size):
    #step = HIDDEN_NODES +1
    #Input train_input shape = [batch_size*step,4]

    #return embed_v  shape = [batch_size*Hidden_nodes,feature_interval]
    #       embed_v_sd shape = embed_v.shape
    #       embed_v_avg shape = embed_v.shape

    #计算每个点的速度
    embed_v = np.zeros([batch_size*HIDDEN_NODES,1],dtype=np.float32)
    step = HIDDEN_NODES+1
    for i in range(batch_size):
        #u=0
        for j in range(step):
            if j == HIDDEN_NODES:
                continue
            dis = util.jwd2dis(train_input[i*step+j][0],
                               train_input[i*step+j][1],
                               train_input[i*step+j+1][0],
                               train_input[i*step+j+1][1])
            embed_v[i*HIDDEN_NODES+j][0] = dis/((train_input[i*step+j+1][3] - train_input[i*step+j][3])*3600*24)


    #计算平均速度
    embed_v_avg = np.zeros([batch_size*HIDDEN_NODES,1],dtype=np.float32)
    for i in range(batch_size):
        start = i*HIDDEN_NODES
        end = start + HIDDEN_NODES
        embed_v_avg[start:end]= np.mean(embed_v[start:end,:])

    #计算每个点的速度偏差
    embed_v_sd = np.zeros(embed_v.shape,dtype=np.float32)
    for i in range(batch_size):
        start = i * HIDDEN_NODES
        end = start + HIDDEN_NODES
        embed_v_sd[start:end] = np.std(embed_v[start:end])


    #将每个特征划分为20个区间首先是将最大速度设为40
    #每个点的速度嵌入onehot向量
    embed_v = np.floor(embed_v/5)
    embed_v = np.array(embed_v,dtype=np.int32)
    embed_v[embed_v > 19] = 19
    embed_v = np.array(trajectory_data.dense_to_one_hot(embed_v,FEATURE_INTERVALS),dtype=np.float32)

    #速度平均值嵌入onehot向量
    embed_v_avg = np.floor(embed_v_avg/5)
    embed_v_avg = np.array(embed_v_avg,dtype=np.int32)
    embed_v_avg[embed_v_avg > 19] = 19
    embed_v_avg = np.array(trajectory_data.dense_to_one_hot(embed_v_avg,FEATURE_INTERVALS),dtype=np.float32)

    #速度偏差嵌入onehot向量
    #embed_v_sd = np.round(embed_v_sd/2)
    #embed_v_sd = embed_v_sd + 10
    embed_v_sd = np.floor(embed_v_sd*2 )
    embed_v_sd = np.array(embed_v_sd,dtype=np.int32)
    embed_v_sd[embed_v_sd > 19] = 19
    embed_v_sd = np.array(trajectory_data.dense_to_one_hot(embed_v_sd,FEATURE_INTERVALS),dtype=np.float32)


    return embed_v,embed_v_sd,embed_v_avg


def embedding(embed_before_v,embed_before_v_sd,embed_before_v_avg,weights,biases):
    #embed_v shape = [batch_size * step , 50]
    embed_v = tf.matmul(embed_before_v,weights["embed_weight_V"],name="embed_v_matmul_embed_before_v") + biases["embed_biases_V"]
    # embed_v shape = [batch_size * step , 50]
    embed_v_sd = tf.matmul(embed_before_v_sd,weights["embed_weight_Vsd"],name="embed_v_sd_matmul_embed_before_v_sd")+biases["embed_biases_Vsd"]
    # embed_v shape = [batch_size* step , 50]
    embed_v_avg = tf.matmul(embed_before_v_avg,weights["embed_weight_Vavg"],name="embed_v_avg_matmul_embed_before_v_avg") + biases["embed_biases_Vavg"]

    #先加v与Vsd
    #result = embed_v + embed_v_sd

    util.variable_summaries(embed_v)
    util.variable_summaries(embed_v_sd)
    util.variable_summaries(embed_v_avg)

    result = tf.add_n([embed_v,embed_v_sd,embed_v_avg])


    result = tf.reshape(result,[-1,HIDDEN_NODES,EMBEDDING_DIMS])
    return result


def bidi_gru(X):


    #第一层双向gru
    l1_f_cell = tf.nn.rnn_cell.GRUCell(HIDDEN_NODES, name="l1_f_gru_cell", activation=maxout_activator)
    l1_b_cell = tf.nn.rnn_cell.GRUCell(HIDDEN_NODES, name="l1_b_gru_cell", activation=maxout_activator)

    l1_f_state = l1_f_cell.zero_state(BATCH_SIZE,dtype=tf.float32)
    l1_b_state = l1_b_cell.zero_state(BATCH_SIZE,dtype=tf.float32)

    #返回结果，返回的结果为正向与反向的output与hstate
    result_1 = tf.nn.bidirectional_dynamic_rnn(l1_f_cell,l1_b_cell,X,initial_state_fw=l1_f_state,initial_state_bw=l1_b_state)

    # 第二层双向gru
    l2_f_cell = tf.nn.rnn_cell.GRUCell(HIDDEN_NODES, name="l2_f_gru_cell", activation=maxout_activator)
    l2_b_cell = tf.nn.rnn_cell.GRUCell(HIDDEN_NODES, name="l2_b_gru_cell", activation=maxout_activator)

    l2_f_state = l2_f_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
    l2_b_state = l2_b_cell.zero_state(BATCH_SIZE, dtype=tf.float32)

    #0 outputs: A tuple (output_fw, output_bw)
    #l1_output_fw = result_1[0][0]
    #l1_output_bw = result_1[0][1]

    #l2_input = result_1[0][0] + result_1[0][1] #tf.concat(result_1[0],2)

    result_2_f = tf.nn.dynamic_rnn(l2_f_cell,result_1[0][0],initial_state=l2_f_state)
    result_2_b = tf.nn.dynamic_rnn(l2_b_cell,result_1[0][1],initial_state=l2_b_state)

    #result_2 = tf.nn.bidirectional_dynamic_rnn(l2_f_cell,l2_b_cell,l2_input,initial_state_fw=l2_f_state,initial_state_bw=l2_b_state)

    util.variable_summaries(l1_f_state)
    util.variable_summaries(l1_b_state)
    util.variable_summaries(l2_f_state)
    util.variable_summaries(l2_b_state)
    #只返回outputs
    return (result_2_f[0],result_2_b[0])


def output_layer(inputs,weight,biase):
    #inputs 为gruoutputs shape = [batch_size,step,num_units]
    output = inputs[0] + inputs[1]
    output = tf.reshape(output,[-1,HIDDEN_NODES])

    #result shape = [batch_size * step,Hidden_node]
    result = tf.matmul(output,weight) + biase
    y = tf.nn.softmax(result,name="sofmax")
    util.variable_summaries(weight)
    util.variable_summaries(biase)

    return y


def train_step(y,y_):

    #y_ = tf.placeholder(tf.float32,shape=[None,1])

    adam = tf.train.AdamOptimizer(0.01)
    cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)),name="cross_emtropy")
    tf.summary.scalar("loss",cross_entropy)

    return adam.minimize(cross_entropy)


def accuracy_function(y,y_):

    # 精度
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    tf.summary.scalar("accuracy",accuracy)
    return accuracy

#embedding层权重
#将20维德向量扩展到50维
embed_weights = {

    "embed_weight_V" : tf.Variable(tf.random_uniform([FEATURE_INTERVALS,EMBEDDING_DIMS],0,0.001,dtype=np.float32),name="embed_weight_V"),
    "embed_weight_Vavg" : tf.Variable(tf.random_uniform([FEATURE_INTERVALS,EMBEDDING_DIMS],0,0.001,dtype=np.float32),name="embed_weight_Vavg"),
    "embed_weight_Vsd" : tf.Variable(tf.random_uniform([FEATURE_INTERVALS,EMBEDDING_DIMS],0,0.001,dtype=np.float32),name="embed_weight_Vsd")

}

embed_biases = {
    "embed_biases_V" : tf.Variable(tf.constant(0.001,shape=[EMBEDDING_DIMS,],dtype=np.float32),name = "embed_biases_V"),
    "embed_biases_Vavg" : tf.Variable(tf.constant(0.001,shape=[EMBEDDING_DIMS,],dtype=np.float32),name= "embed_biases_Vavg"),
    "embed_biases_Vsd" : tf.Variable(tf.constant(0.001,shape=[EMBEDDING_DIMS,],dtype=np.float32),name="embed_biases_Vsd")

}

#输出层权重
out_weight = tf.Variable(tf.random_uniform([HIDDEN_NODES,MODE],0,0.001,dtype=np.float32),name = "out_weight")
out_biase = tf.Variable(tf.constant(0.001,shape=[MODE,],dtype=np.float32),name = "out_biase")

#x = tf.placeholder(tf.float32,shape=[None,])
#y_ = tf.placeholder(tf.float32,shape=[None,MODE])

embed_b_v_p = tf.placeholder(tf.float32,shape=[None,FEATURE_INTERVALS],name="embed_b_v_p")
embed_b_v_sd_p = tf.placeholder(tf.float32,shape=[None,FEATURE_INTERVALS],name="embed_b_v_sd_p")
embed_b_v_avg_p = tf.placeholder(tf.float32,shape=[None,FEATURE_INTERVALS],name="embed_b_v_avg_p")
y_p = tf.placeholder(tf.float32,shape=[None,MODE],name="y_p")

embed_result = embedding(embed_b_v_p, embed_b_v_sd_p, embed_b_v_avg_p,embed_weights,embed_biases)
gru_output = bidi_gru(embed_result)
pred = output_layer(gru_output,out_weight,out_biase)
train_op = train_step(pred,y_p)

accuracy = accuracy_function(pred,y_p)


init = tf.global_variables_initializer()



with tf.Session() as sess:
    #sess = tfdbg.LocalCLIDebugWrapperSession(sess)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
    train_data_set = trajectory_data.read_data_set("data/","geo_train_data_4_mode.txt",BATCH_SIZE,HIDDEN_NODES)
    # train_data_set.struct_train_data(ITER_NUM)
    # labels = train_data_set.labels()
    # #归一化
    # embed_b_v_all,embed_b_v_avg_all,embed_b_v_sd_all = normalization(train_data_set.datas())



    sess.run(init)
    for i in range(ITER_NUM):
        #start = i*BATCH_SIZE*HIDDEN_NODES
        #end = start + BATCH_SIZE*HIDDEN_NODES
        batch_x,batch_y = train_data_set.next_batch(BATCH_SIZE)
        batch_y = np.array(batch_y, dtype=np.int32)
        y_ = trajectory_data.dense_to_one_hot(batch_y, MODE)

        #embed_b_v, embed_b_v_sd, embed_b_v_avg = to_onehot(embed_b_v_all[start:end],embed_b_v_avg_all[start:end],embed_b_v_sd_all[start:end])

        embed_b_v, embed_b_v_sd, embed_b_v_avg =  embedding_before(batch_x,BATCH_SIZE)

        summary,tt = sess.run([merged,train_op],feed_dict={embed_b_v_p:embed_b_v,
                                  embed_b_v_avg_p:embed_b_v_avg,
                                  embed_b_v_sd_p:embed_b_v_sd,
                                  y_p:y_})
        train_writer.add_summary(summary)

        if i%20 == 0:

            print(sess.run(accuracy,feed_dict={embed_b_v_p:embed_b_v,
                                  embed_b_v_avg_p:embed_b_v_avg,
                                  embed_b_v_sd_p:embed_b_v_sd,
                                  y_p:y_}))
            yy = (np.reshape(batch_y, [-1]))
            print(yy[np.arange(BATCH_SIZE)*HIDDEN_NODES])


    train_writer.close()