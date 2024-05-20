from django.urls import path

from . import views
from django.views.static import serve
# 设置命名空间，防止多个app中的url重名
# 访问view的规则： app:index
app_name = 'app'

urlpatterns = [
    path("index", views.index, name="index"),

    path("glove", views.glove, name="glove"),
    path("glove/state", views.state, name="state"),
    path("glove/adduser", views.adduser, name="adduser"),
    path("glove/getdata", views.getdata, name="getdata"),
    path("glove/result", views.result, name="result"),
    path("glove/stop", views.stop, name="stop"),
    path("glove/train", views.train, name="train"),
    path("glove/setname", views.setname, name="setname"),
    path("glove/setlabel", views.setlabel, name="setlabel"),
    path("glove/savedata", views.savedata, name="savedata"),
    path("glove/train_model", views.train_model, name="train_model"),
    path("glove/transfer_data", views.transfer_data, name="transfer_data"),
    path("glove/update_acc", views.update_acc, name="update_acc"),
    path("glove/get_ten_acc", views.get_ten_acc, name="get_ten_acc"),
]