# -*- coding: utf-8 -*-
"""
Created on Sat May  7 15:01:42 2022

@author: Asus
"""

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential #Kullanaccak model
from keras.layers import Dense, Activation, Dropout #katmanlarımız
from tensorflow.keras.optimizers import SGD
import random

kelimeler=[]
siniflar = []
belgeler = []
ignore_words = ['?', '!',"."]
data_file = open('intents.json', encoding='utf-8').read()
intents = json.loads(data_file, strict=False)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        #kelimeleri tokenize et
        w = nltk.word_tokenize(pattern)
        kelimeler.extend(w)
        # gövdede belge ekle
        belgeler.append((w, intent['tag']))

        #sınıflar listesine eklenir
        if intent['tag'] not in siniflar:
            siniflar.append(intent['tag'])

# lemmatize işlemi ile anlamlı kökler elde edilir ve kopyalar kaldırılır.
kelimeler = [lemmatizer.lemmatize(w.lower()) for w in kelimeler if w not in ignore_words]
kelimeler = sorted(list(set(kelimeler)))
# sınıfları sırala
siniflar = sorted(list(set(siniflar)))
# belgeler = patterns ve intents arasındaki kombinasyon
print (len(belgeler), "belgeler")
# sınıflar = intents
print (len(siniflar), "sınıflar", siniflar)
# kelimeler = tüm kelimeler, kelime bilgisi
print (len(kelimeler), "eşsiz lemmatized kelimeler", kelimeler)


pickle.dump(kelimeler,open('kelimeler.pkl','wb'))
pickle.dump(siniflar,open('sınıflar.pkl','wb'))

# eğitim verilerimizi oluşturuyoruz
training = []
# çıktı için boş bir dizi
output_empty = [0] * len(siniflar)
# eğitim seti, her cümle için kelimelerle dolu "bag" oluşturuyoruz.
for doc in belgeler:
    # başlat--> bag of words
    bag = []
    # pattern için tokenized edilmiş kelimelerin listesi
    pattern_words = doc[0]
    # her kelimeyi lemmatize et
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # mevcut pattern'de kelime eşleşmesi bulunursa, 1 ile dizimizi oluşturur
    for w in kelimeler:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # çıktı her tag için '0' ve  geçerli etiket için (her pattern için) '1' 
    output_row = list(output_empty)
    output_row[siniflar.index(doc[1])] = 1
    
    training.append([bag, output_row])
# özelliklerimizi karıştır ve np.array'e çevir
random.shuffle(training)
training = np.array(training)
#  train and test listeleri oluştur. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("******************* EĞİTİM VERİLERİ OLUŞTURULDU *******************")


# Model oluştur - 3 katman. Birinci katman 128 nöron, ikinci katman 64 nöron 
# ve üçüncü çıktı katmanı nöron sayısını içerir.
# softmax ile tahmin edilen çıkıtı niyet sayısı, intents sayısına eşit

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Modeli derle. Stochastic gradient descent with Nesterov accelerated gradient
#  bu model için iyi sonuçlar veriyor.
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting ve modeli kaydet
hist = model.fit(np.array(train_x), np.array(train_y), epochs=250, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("*******************  MODEL OLUŞTURULDU *******************")