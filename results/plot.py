import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas
model_name = 'SeqCapsGAN'
model_title = 'Caps'
pos = -6

acc = pandas.read_csv('./saved_models/'+model_name+'/accs.csv')
loss = pandas.read_csv('./saved_models/'+model_name+'/losses.csv')


# Create some mock data
acc_train = acc['train_accs']
acc_test = acc['test_accs']
loss_train = loss['train_losses']
loss_test = loss['test_losses']

t = range(len(acc_train))

mpl.style.use('seaborn-paper')

fig = plt.figure()
ax = fig.add_subplot(111)
lns1 = ax.plot(t, acc_train, 'tab:orange', label = 'train acc')
lns2 = ax.plot(t, acc_test, 'tab:blue', label = 'valid acc')
ax2 = ax.twinx()
lns3 = ax2.plot(t, loss_train, 'tab:red', label = 'train loss')
lns4 = ax2.plot(t, loss_test, 'tab:green', label = 'valid loss')

# added these three lines
lns = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=4, fancybox=True)

ax.grid()
ax.set_xlabel("epochs")
ax.set_ylabel("accuracy %")
ax2.set_ylabel("loss")
ax.text(pos, 113, model_title, fontsize=11)
plt.savefig(model_name+'.pdf')
plt.show()