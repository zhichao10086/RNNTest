import numpy as np
import os
import re
import pandas as pd

data_dir = "G:/新建文件夹/Geolife Trajectories 1.3/Data"

def find_labels_datadir(path):
    dirlist = os.listdir(data_dir)
    dirs = []

    for i in dirlist:
        path = data_dir + "/" + i + "/labels.txt"
        if os.path.exists(path):
            dirs.append(i)

    return dirs

#将交通方式字符串替换成数字
def switch_mode(str):
    str = str.strip()
    if(str == "bike"):
        return "0"

    if(str == "car"):
        return "1"

    if (str == "walk"):
        return "2"

    if (str == "bus"):
        return "3"

    if (str == "train"):
        return "4"

    if (str == "subway"):
        return "5"

    if (str == "airplane"):
        return "6"

    if (str == "taxi"):
        return "7"
    if (str == "boat"):
        return "8"
    if (str == "run"):
        return "9"
    if (str == "motorcycle"):
        return "10"
    else:
        print(str)
        return "11"

def sovle_data(mode):
    with open("data/geo_train_data.txt") as f:
        f.readline()
        fw = open("data/geo_train_data_4_mode.txt", "w")
        temp = "0"
        for line in f.readlines():
            li = re.split(" ", line)
            #print(line[3])
            if(li[3] == temp):
                continue

            if(line[len(line)-2] >= mode):
                continue
            else:
                temp = li[3]
                fw.write(line)
        fw.close()


def structure_data1():
    with open("data/Geolife_tra_label.txt") as f:
        f.readline()
        fw = open("data/geo_train_data.txt", "w")

        for line in f.readlines():
            li = re.split("\t|\n|\"", line)
            # ['', '180199', '', '', '010', '', '', '39.894178', '', '', '116.3182', '', '', '-777', '', '', '39535.6212962963', '', '', '2008/3/28', '', '', '14:54:40', '', '', '2', '', '', 'train', '', '']
            list = []
            for i in [7, 10, 13, 16]:
                list.append(li[i])
                list.append(" ")
            list.append(switch_mode(li[28]))

            fw.write(''.join(list))
            fw.write("\n")
        fw.close()



def structure_data():
    #dataset = np.loadtxt("data/Geolife_tra_label.txt")
    with open("data/Geolife_tra_label.txt")as f:
        #跳过开头
        f.readline()
        fw = open("data/geo_train_data.txt","w")

        for i in range(100000):
            line = f.readline()
            li = re.split("\t|\n|\"",line)
            #['', '180199', '', '', '010', '', '', '39.894178', '', '', '116.3182', '', '', '-777', '', '', '39535.6212962963', '', '', '2008/3/28', '', '', '14:54:40', '', '', '2', '', '', 'train', '', '']
            list = []
            for i in [7,10,13,16]:
                list.append(li[i])
                list.append(" ")
            list.append(switch_mode(li[28]))

            fw.write(''.join(list))
            fw.write("\n")
        fw.close()

        fwtest = open("data/geo_test_data.txt","w")

        for i in range(20000):
            line = f.readline()
            li = re.split("\t|\n|\"", line)
            # ['', '180199', '', '', '010', '', '', '39.894178', '', '', '116.3182', '', '', '-777', '', '', '39535.6212962963', '', '', '2008/3/28', '', '', '14:54:40', '', '', '2', '', '', 'train', '', '']
            list = []
            for i in [7, 10, 13, 16]:
                list.append(li[i])
                list.append(" ")
            list.append(switch_mode(li[28]))

            fwtest.write(''.join(list))
            fwtest.write("\n")

        fwtest.close()


#注意数据类型，int与float数组长度不一样
def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


class Dataset(object):

    def __init__(self,data,labels,batch_size,num_step):
        self._data = data
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = data.shape[0]
        self._num_step = num_step
        self._train_data =None
        self._label_data = None
        self._index = 0
        self.batch_size = batch_size

    def datas(self):
        return self._data

    def labels(self):
        return self._labels

    def get_train_data(self):

        #self.struct_train_data()
        return self._train_data

    def struct_train_data(self,n):
        #将所有数据筛选出来训练数据
        self._train_data,self._label_data = self.next_batch(self.batch_size)

        for i in range(1,n):
            train_data,label_data = self.next_batch(self.batch_size)
            np.vstack((self._train_data,train_data))
            np.vstack((self._label_data,label_data))



    #从有用的Train_data中选出下一批数据
    def next_batch_1(self):

        start = self._index
        end = start + self.batch_size * (self._num_step + 1)

        self._index = end + 1
        return self._train_data[start:end],self._label_data[start:end]



    def next_batch(self, batch_size):
        #从开始 每num_step为一组数据，

        start  = self._index_in_epoch
        end = 0
        self._index_in_epoch += batch_size
        #if start + batch_size > self._num_examples:
        #   self._epochs_completed += 1

        temp_batch_size = 0
        train_data = np.zeros([batch_size * (self._num_step+1),self._data.shape[1]])
        label_data = np.zeros([batch_size * self._num_step,self._labels.shape[1]])
        #label_result = np.zeros([batch_size*(self._num_step-1),self._labels.shape[1]])

        while(temp_batch_size < batch_size  ):

            if(start + self._num_step+1 >self._num_examples):
                #print("start",start,self._index_in_epoch)
                self._index_in_epoch = np.random.randint(0,10,1)[0]
                start = self._index_in_epoch
            #要26个数据
            end = start + self._num_step+1
            label = self._labels[start:end,:]
            result = (label == label[0])
            #查看是否标签全是相同的
            if(False in result):
                #print("error")
                start += 5
                continue
            else:

                train_r_start = temp_batch_size*(self._num_step+1)
                train_r_end = (temp_batch_size+1)*(self._num_step+1)
                label_r_start = temp_batch_size*self._num_step
                label_r_end = (temp_batch_size+1)*self._num_step

                #temp_train_data[r_start:r_end,:] = self._data[start:end,:]
                temp_train = self._data[start:end,:]

                #判断如果时间有相等的就放弃这组数据
                temp_time = temp_train[:,3]
                temp_time = np.reshape(temp_time,-1)
                time_s = pd.Series(temp_time)
                if not time_s.is_unique:
                    start += 5
                    continue

                flag = 0
                for i in range(self._num_step+1):
                    if i == self._num_step :
                        continue
                    t = (temp_train[i+1][3] - temp_train[i][3])*3600*24
                    if t > 120:
                        flag = 1
                        break
                #当时间过渡太长的话 取消这组数据
                if flag==1:
                    start += 5
                    continue

                train_data[train_r_start:train_r_end, :] = self._data[start:end, :]
                label_data[label_r_start:label_r_end,:] = self._labels[start:end-1,:]
                temp_batch_size += 1
                start += self._num_step

        self._index_in_epoch = end + 1


        return train_data,label_data






def read_data_set(datadir,txtname,batch_size,num_step):

    train_data_txt = datadir + txtname
    train_data = np.loadtxt(train_data_txt)
    #切割数据
    #train_data_set = Dataset(train_data[0:10000,:-1],train_data[0:10000,train_data.shape[1]-1 :train_data.shape[1]])
    train_data_set = Dataset(train_data[:,:-1],train_data[:,train_data.shape[1]-1 :train_data.shape[1]],batch_size=batch_size,num_step = num_step)
    return train_data_set

if __name__ == "__main__":
    sovle_data("4")
