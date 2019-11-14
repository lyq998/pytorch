import pickle

path = 'D:/python/pytorch/back_up/new_run/offspringdata/'
path1 = path+'gen_1.dat'
path2 = path+'gen_2.dat'
with open(path1, 'rb') as file_handler:
    data = pickle.load(file_handler)
    print(data['gen_no'], data['pops'])
with open(path2, 'rb') as file_handler:
    data = pickle.load(file_handler)
    print(data['gen_no'], data['pops'])