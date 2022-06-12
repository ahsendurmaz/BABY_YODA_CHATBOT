# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:40:11 2022

@author: Asus
"""

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np


from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('kelimeler.pkl','rb'))
classes = pickle.load(open('sınıflar.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - kelimeler diziye ayrılır
    sentence_words = nltk.word_tokenize(sentence)
    # kelimelerin kökleri - kelimeler için kısa form
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# kelime dizisinin dönüş torbası: Cümlede bulunan torbadaki her kelime için 0 veya 1

def bow(sentence, words, show_details=True):
    # kalıp belirtilir
    sentence_words = clean_up_sentence(sentence)
    # kelime torbası - N kelimelik matris, kelime matrisi
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # Aktif(geçerli) kelime, kelime konumundaysa 1 atanır.
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # threshold(eşik) altındaki tahminleri filtrele
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # olabilirliğine göre sıralar
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res

#Tkinter kullanarak grafiksel arayüzü (GUI) tasarlama
import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "Siz: " + msg + '\n\n')
        ChatLog.config(foreground="black", font=("Verdana", 12 ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Chatbot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

base = Tk()
base.title("BABY CHATBOT")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)
base.iconbitmap("icon.ico")

#Sohbet Penceresi oluşturma
ChatLog = Text(base, bd=0, bg= "#DBD4E8", height="8", width="50", font="Arial",cursor="star")

ChatLog.config(state=DISABLED)

#scrollbar, Chat windowa bağlanır
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="star")
ChatLog['yscrollcommand'] = scrollbar.set

#Gönderme butonu oluşturma
SendButton = Button(base, font=("Verdana",12,'bold'), text="Gönder", width="12", height=5,
                    bd=0, bg="#968BAF", activebackground="#C6BCDB",fg='#ffffff',
                    command= send )

#Mesajın yazılacağı kutuyu oluşturma
EntryBox = Text(base, bd=0, bg="#DBD4E8",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Tüm bileşenleri ekrana yerleştirme
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
