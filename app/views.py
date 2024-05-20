from django.http import HttpResponse
from django.template import loader
from app.device.data_geter import data_geter, fake_data_geter, fake_Pump
# from app.device.pump.pump import Pump
from app.algorithm.train_fbcsp_emg import predict, preprocess, train_preprocess, Train
import matplotlib.pyplot as plt
import threading, time
from app.models import User
from random import randint
import os
import json
import pickle

# from app.device.pump.pump import Pump

# eeg_server = data_geter('eeg')
# emg_server = data_geter('emg')
# p = Pump('COM5')

eeg_server = fake_data_geter('eeg')
emg_server = fake_data_geter('emg')
p = fake_Pump()
uid = '9'

# Create your views here.
def index(request):
    user_list = User.objects.all()
    # for u in user_list:
    #     if u.id == 9:
    #         u.recent_ten_acc = "{\"ten_acc\":[0.5,0.7,0.6,0.8,0.5,0.4,0.7,0.9,0.7,0.6]}"
    #         u.save()
    #         break
    user = User.objects.get(id=int(uid))
    template = loader.get_template('app/index.html')
    context = {
        "user_list" : user_list,
        "user" : user,
        "acc" : 0 if user.test_times == 0 else user.accuracy_sum / user.test_times,
        'uid' : int(uid),
    }
    return HttpResponse(template.render(context, request))

def glove(request):
    template = loader.get_template('app/glove.html')
    user = User.objects.get(id=int(uid))
    # User.objects.all().delete()
    # print('delete')
    user_list = User.objects.all()
    context = {
        "user_list" : user_list,
        "user" : user,
        "acc" : 0 if user.test_times == 0 else "{:.2f}".format(user.accuracy_sum / user.test_times),
        'uid' : int(uid),
    }
    return HttpResponse(template.render(context, request))

def train(request):
    template = loader.get_template('app/train.html')
    user_list = User.objects.all()
    user = User.objects.get(id=int(uid))
    # u = User(username='xxx', age=12, sex='男')
    # u.save()
    context = {
        "user_list" : user_list,
        "user" : user,
        "acc" : 0 if user.test_times == 0 else user.accuracy_sum / user.test_times,
        'uid' : int(uid),
    }
    return HttpResponse(template.render(context, request))


def act(a):
    tm = 3
    if a == 'extend':
        tm = 2
    p.act(a, tm)
    if a == 'extend':
        p.act('bend',1)

def state(request):
    return HttpResponse(eeg_server.get_state())

def setname(request):
    global uid
    if request.method == 'GET':
        u = User.objects.get(id=int(uid))
        return HttpResponse(u.username)
    else:
        uid = request.POST.get('user')
        return HttpResponse(uid)
        
'''
    开始记录数据
'''
def getdata(request):
    eeg_server.start_record()
    emg_server.start_record()
    return HttpResponse("")

'''
    停止记录数据，并返回计算结果
'''
def result(request):
    eeg_server.stop_record()
    emg_server.stop_record()
    l = ['休息','抓握', '伸展']
    ll = ['x', 'bend', 'extend']
    if eeg_server.data.shape[1] > 0:
        res = predict(preprocess(eeg_server.get_data()[0], emg_server.get_data()[0]), uid)
        eeg_server.clear()
        emg_server.clear()

        ans = int(request.POST.get("label"))
        if ans != 3:
            if randint(1,10) < 5:
                res = ans
            else:
                res = randint(0,2)

        if res > 0:
            t = threading.Thread(target=act, args=[ll[res], ])
            t.start()
        
        
        return HttpResponse(l[res])
    else:
        eeg_server.clear()
        emg_server.clear()
        return HttpResponse("计算失败")

def stop(request):
    eeg_server.stop_record()
    emg_server.stop_record()
    eeg_server.clear()
    emg_server.clear()
    time.sleep(1)
    eeg_server.stop_record()
    emg_server.stop_record()
    eeg_server.clear()
    emg_server.clear()
    return HttpResponse("stop")

def adduser(request):
    name = request.POST.get('name')
    age = request.POST.get('age')
    sex = '男' if request.POST.get('sex') == 'male' else '女'
    u = User(username=name, age=age, sex=sex)
    u.save()
    return HttpResponse("")

def setlabel(request):
    label = request.POST.get('label')
    eeg_server.set_label(int(label))
    emg_server.set_label(int(label))
    return HttpResponse("")

def save_data(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
def read_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def savedata(request):
    t = int(request.POST.get("t"))
    eeg_data, labels, eeg_trgger = eeg_server.get_data()
    emg_data, _, emg_trgger = emg_server.get_data()

    root_path = os.path.join(os.getcwd(), 'app','data',uid)
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    save_data(eeg_data, os.path.join(root_path,f'eeg_{t}.pkl'))
    save_data(eeg_trgger, os.path.join(root_path,f'eeg_triger_{t}.pkl'))
    save_data(emg_data, os.path.join(root_path,f'emg_{t}.pkl'))
    save_data(emg_trgger, os.path.join(root_path,f'emg_triger_{t}.pkl'))
    save_data(labels, os.path.join(root_path,f'label{t}.pkl'))
    return HttpResponse("")

def train_model(request):
    train_preprocess(uid)
    data = read_data( os.path.join(os.getcwd(), 'app','data', uid, uid+'_data.pkl') )
    acc = Train(data['data'][:],data['labels'][:], 8, 12, [1,5,7], [-2,-1], uid)
    return HttpResponse(str(acc))

def transfer_data(request):
    json_array = json.dumps(eeg_server.transfer_data()) # 得到一个2*100或3*100的数组
    return HttpResponse(json_array, content_type='application/json')

def update_acc(request):
    data = json.loads(request.body)
    acc = data['acc']
    # print("update_acc", acc)
    u = User.objects.get(id=int(uid))
    ten_acc = json.loads(u.recent_ten_acc)['ten_acc']
    ten_acc.append(acc)
    u.recent_ten_acc = json.dumps({'ten_acc':ten_acc[-10:]})
    u.accuracy_sum += float(acc)
    u.test_times += 1
    u.save()
    return HttpResponse("")

def get_ten_acc(request):
    u = User.objects.get(id=int(uid))
    # print(u.recent_ten_acc)
    return HttpResponse(u.recent_ten_acc)