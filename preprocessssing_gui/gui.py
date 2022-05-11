# libraries 

import tkinter
from no_gui import *

# properities
width=40
border = 4
color ='Red'
height=3
font=("Arial", 14)
#-----------------------------------------------

top=tkinter.Tk()
top.title("Spam Steps")
top.geometry("1000x500")
labelframe=tkinter.LabelFrame(top,text='Here you can see output sample ' ,font=("Arial", 20))
top["bg"] = "#98AFC7"
labelframe["bg"] = "#98AFC7"
#-----------------------------------------------
# methods
path = 'data_set.csv'
copy = read_data_set(path)
data_copy = Cleaning(copy)

def tokenize():
    # print(data_copy.head())
    tokenization(data_copy)
    sampleOfOutput = data_copy['tokens'][0]
    tkinter.Label.config(label,text=f'{sampleOfOutput}')
def segment():
    segmentation(data_copy)
    sampleOfOutput= data_copy['sentenses'][0]
    tkinter.Label.config(label,text=f'{sampleOfOutput}')
def stemming_g():
    stemming(data_copy)
    sampleOfOutput= data_copy['root_stem'][0]
    tkinter.Label.config(label,text=f'{sampleOfOutput}')
def stemming_Light_g():
    stemming_Light(data_copy)
    sampleOfOutput= data_copy['light_stem'][0]
    tkinter.Label.config(label,text=f'{sampleOfOutput}')
def drop_stop_words_g():
    drop_stop_words(data_copy)
    sampleOfOutput= data_copy['without_stopwords'][0]
    tkinter.Label.config(label,text=f'{sampleOfOutput}')
#----------------------------------------------
but_1=tkinter.Button(top,text='Tokenize',command=tokenize,
        width=width,background=color,border=border,height=height)

but_2=tkinter.Button(top,text='Segmentation',command=segment
        ,width=width,border=border,height=height,background=color)
        
but_3=tkinter.Button(top,text='Stemming',command=stemming_g
        ,width=20,border=border,height=height,background=color)

but_3_=tkinter.Button(top,text='Stemming_Light',command=stemming_Light_g
        ,width=20,border=border,height=height,background=color)

but_4=tkinter.Button(top,text='Drop_stop_words',command=drop_stop_words_g
        ,width=width,border=border,height=height,background=color)

label=tkinter.Label(labelframe,text='output',font=font,background="#98AFC7")
#-----------------------------------------------

but_1.place(relx=0,rely=.1)
but_2.place(relx=.7,rely=.1)
but_3.place(relx=0,rely=.3)
but_3_.place(relx=.16,rely=.3)
but_4.place(relx=.7,rely=.3)
labelframe.place(x=0,y=250,height=250,width=1000)
label.grid()
top.mainloop()

