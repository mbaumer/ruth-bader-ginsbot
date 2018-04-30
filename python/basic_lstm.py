from __future__ import division
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM, Bidirectional
from sklearn.metrics import roc_curve

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import nltk
try: 
    import vocab
except ImportError:
    import sys
    sys.path.insert(0, '/Users/mbaumer/side_projects/ruth-bader-ginsbot/python/')
    import vocab
import keras

data = pd.read_csv('../data/supreme_court_dialogs_corpus_v1.01/supreme.conversations.txt',sep='\ \+\+\+\$\+\+\+\ ',
            names=['case_id','utterance_id','after_previous','speaker','isJustice','justice_vote','presenting_side','utterance'])

justice_lines = data[(data['isJustice'] == 'JUSTICE') & ((data['justice_vote'] == 'PETITIONER') | (data['justice_vote'] == 'RESPONDENT'))]

tokens = map(nltk.word_tokenize,justice_lines['utterance'])

emb_matrix, word2id, id2word = vocab.get_glove('../data/glove/glove.6B.50d.txt',50)

N_words = 0
N_unk = 0
list_list_tokens = []
for sentence in tokens:
    list_tokens = []
    for word in sentence:
        N_words += 1
        token_id = word2id.get(word,1)
        list_tokens.append(token_id)
        if token_id == 1:
            N_unk += 1
    list_list_tokens.append(list_tokens)
print ('Nwords:', N_words)
print ('Nunk:', N_unk)
print ('%unk:', N_unk/N_words*100)

y = np.zeros(len(justice_lines))
y[np.where(justice_lines['justice_vote'] == 'PETITIONER')] = 1

shuffled = np.random.permutation(np.arange(len(justice_lines),dtype=int))
train_inds = shuffled[:int(len(justice_lines)*.7)]
test_inds = shuffled[int(len(justice_lines)*.7):]

max_features = len(word2id)
maxlen = 200
batch_size = 128

print('Loading data...')
x_train = [list_list_tokens[train_ind] for train_ind in train_inds]
y_train = y[train_inds]
x_test = [list_list_tokens[test_ind] for test_ind in test_inds]
y_test = y[test_inds]
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2)))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=20,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

y_pred = model.predict(x_train)
y_pred_test = model.predict(x_test)

fpr,tpr, _ = roc_curve(y_train,y_pred)
fpr1,tpr1, _ = roc_curve(y_train,np.ones_like(y_train))
fpr2,tpr2, _ = roc_curve(y_test,y_pred_test)


plt.plot(fpr,tpr,label='LSTM Train')
plt.plot(fpr1,tpr1,label='Always say petitioner')
plt.plot(fpr2,tpr2,label='LSTM Test')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend()
plt.savefig('../results/roc_utterance_level.png')

plt.figure()
plt.hist(y_pred,normed=True,bins=50)
plt.xlabel('y_pred');
plt.vlines(np.sum(y)/len(y),0,5,label='Prob. petitioner wins')
plt.legend()
plt.savefig('../results/ypred_utterance_level.png')

out = justice_lines.iloc[train_inds]
out['pred'] = pd.Series(y_pred.flatten(),index=out.index)
stats = out[['case_id','speaker','justice_vote','pred']].groupby(['case_id','speaker'])
marker = stats.agg({'justice_vote' :'first', 'pred':'mean'})['justice_vote'].values == 'PETITIONER'
ynew = marker.astype('int')
yhat_new = stats.agg({'justice_vote' :'first', 'pred':'mean'})['pred'].values

out = justice_lines.iloc[test_inds]
out['pred'] = pd.Series(y_pred_test.flatten(),index=out.index)
stats = out[['case_id','speaker','justice_vote','pred']].groupby(['case_id','speaker'])
marker = stats.agg({'justice_vote' :'first', 'pred':'mean'})['justice_vote'].values == 'PETITIONER'
ynew_test = marker.astype('int')
yhat_new_test = stats.agg({'justice_vote' :'first', 'pred':'mean'})['pred'].values

fpr3,tpr3, _ = roc_curve(ynew,yhat_new)
fpr4,tpr4, _ = roc_curve(ynew_test,yhat_new_test)
fpr5,tpr5, _ = roc_curve(ynew_test,np.ones_like(ynew_test))
plt.plot(fpr3,tpr3,label='Justice level train')
plt.plot(fpr4,tpr4,label='Justice level test')
plt.plot(fpr5,tpr5,label='Justice level -- say petitioner')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend()
