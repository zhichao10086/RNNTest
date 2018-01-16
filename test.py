from functools import reduce

a = [1,2,3]
b = [1,2,3]


v = list(zip(a,b))

input_vecs = [[1,1], [0,0], [1,0], [0,1]]
    # 期望的输出列表，注意要与输入一一对应
    # [1,1] -> 1, [0,0] -> 0, [1,0] -> 0, [0,1] -> 0
labels = [1, 0, 0, 0]

weights = [0.0,0.0]



samples = zip(input_vecs,labels)
for (input_vec,label) in samples:
    print(reduce(lambda x,y:x+y,map(lambda x:x[0]*x[1],zip(input_vec,weights))))
    print(list(zip(input_vec,weights)))

c = list(zip(input_vecs,labels))

print(reduce(lambda x,y:x+y,a))

#print(list(map(lambda x:x[0],c)))
