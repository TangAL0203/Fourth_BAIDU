#-*-coding:utf-8-*-
import os
import os.path as osp
import torch
from torch.autograd import Variable

def train_batch(model, optimizer, batch, label): 
    optimizer.zero_grad() # 
    input = Variable(batch)
    output = model(input)
    criterion = torch.nn.CrossEntropyLoss()
    criterion(output, Variable(label)).backward() 
    optimizer.step()
    return criterion(output, Variable(label)).data

def train_epoch(model, num_batches, train_loader, print_freq, optimizer=None):
    for batch, label in train_loader:
        loss = train_batch(model, optimizer, batch.cuda(), label.cuda())
        if num_batches%print_freq == 0:
            print('%23s%-9s%-13s'%('the '+str(num_batches)+'th batch, ','loss is: ',str(round(loss[0],8))))
        num_batches +=1
    return num_batches


def get_train_val_acc(model, train_loader, val_loader):
    model.eval()
    train_correct = 0
    train_total = 0

    val_correct = 0
    val_total = 0

    for i, (batch, label) in enumerate(val_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1] # 返回模型预测概率最大的标签
        val_correct += pred_label.cpu().eq(label).sum() # label为torch.LongTensor类型
        val_total += label.size(0)

    for i, (batch, label) in enumerate(train_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1] # 返回模型预测概率最大的标签
        train_correct += pred_label.cpu().eq(label).sum() # label为torch.LongTensor类型
        train_total += label.size(0)

    print("Train Accuracy :"+str(round( float(train_correct) / train_total , 3 )))
    print("Val   Accuracy :"+str(round( float(val_correct) / val_total , 3 )))

    model.train()
    return round( float(val_correct) / val_total , 3 )

def get_train_val_test_acc(model, train_loader, val_loader, test_loader):
    model.eval()
    train_correct = 0
    train_total = 0

    val_correct = 0
    val_total = 0

    test_correct = 0
    test_total = 0

    for i, (batch, label) in enumerate(train_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1] # 返回模型预测概率最大的标签
        train_correct += pred_label.cpu().eq(label).sum() # label为torch.LongTensor类型
        train_total += label.size(0)

    for i, (batch, label) in enumerate(val_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1] # 返回模型预测概率最大的标签
        val_correct += pred_label.cpu().eq(label).sum() # label为torch.LongTensor类型
        val_total += label.size(0)

    for i, (batch, label) in enumerate(test_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1] # 返回模型预测概率最大的标签
        test_correct += pred_label.cpu().eq(label).sum() # label为torch.LongTensor类型
        test_total += label.size(0)

    print("Train Accuracy :"+str(round( float(train_correct) / train_total , 3 )))
    print("Val   Accuracy :"+str(round( float(val_correct) / val_total , 3 )))
    print("Test  Accuracy :"+str(round( float(test_correct) / test_total , 3 )))

    model.train()
    return round( float(val_correct) / val_total , 3 ), round( float(test_correct) / test_total , 3 )

# do TenCrop operatios on test imags
def get_train_val_testTenCrop_acc(model, train_loader, val_loader, test_loader, TenCroptest_loader):
    model.eval()
    train_correct = 0
    train_total = 0

    val_correct = 0
    val_total = 0

    test_correct = 0
    test_total = 0

    TenCroptest_correct = 0
    TenCroptest_total = 0

    for i, (batch, label) in enumerate(train_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1] # 返回模型预测概率最大的标签
        train_correct += pred_label.cpu().eq(label).sum() # label为torch.LongTensor类型
        train_total += label.size(0)

    for i, (batch, label) in enumerate(val_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1] # 返回模型预测概率最大的标签
        val_correct += pred_label.cpu().eq(label).sum() # label为torch.LongTensor类型
        val_total += label.size(0)

    for i, (batch, label) in enumerate(test_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1] # 返回模型预测概率最大的标签
        test_correct += pred_label.cpu().eq(label).sum() # label为torch.LongTensor类型
        test_total += label.size(0)

    for i, (batch, label) in enumerate(TenCroptest_loader):
        bs, ncrops, c, h, w = batch.size()
        batch = batch.cuda()
        result = model(Variable(batch.view(-1, c, h, w))) # (10,100)
        result_avg = result.view(bs, ncrops, -1).mean(1)
        pred_label = result_avg.data.max(1)[1]
        if i==0:
            print "pred_label is: ", pred_label.cpu()
            print "label is: ", label
        TenCroptest_correct += pred_label.cpu().eq(label).sum()
        TenCroptest_total += label.size(0)


    print("Train         Accuracy :"+str(round( float(train_correct) / train_total , 3 )))
    print("Val           Accuracy :"+str(round( float(val_correct) / val_total , 3 )))
    print("Test          Accuracy :"+str(round( float(test_correct) / test_total , 3 )))
    print("TenCrop Test  Accuracy :"+str(round( float(TenCroptest_correct) / TenCroptest_total , 3 )))

    model.train()
    return round( float(val_correct) / val_total , 3 ), round( float(TenCroptest_correct) / TenCroptest_total , 3 )

