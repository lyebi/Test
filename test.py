# import numpy as np
# # import gym
# # import universe
# # env=gym.make('MontezumaRevenge-v0')
# # state=env.reset()
# # state=np.uint32(state)
# # import copy
# # import cv2
# #
# # cv2.imshow('img',np.uint8(state))
# # cv2.waitKey()
# #
# #
# #
# # for i in range(1000):
# #     a=env.action_space.sample()
# #     tmp,_,_,_=env.step(a)
# #     tmp=np.uint32(tmp)
# #     state+=tmp
# #     state1=np.uint32(state/(i+2))
# #
# #     # cv2.imshow('sad',state)
# #     cv2.imshow('sum',np.uint8(state1))
# #     cv2.waitKey()
#
# import itchat
#
# itchat.login()
# friends=itchat.get_friends(update=True)[0:]
#
# male=female = other=0
#
# for i in friends[1:]:
#     sex=i["Sex"]
#     if sex==1:
#         male+=1
#     elif sex==2:
#         female+=1
#     else:
#         other+=1
#
# total=len(friends[1:])
#
# print('male:',male/total)
# print('female:',female/total)
#
# def get_var(var):
#     variable=[]
#     for i in friends:
#         value=i[var]
#         variable.append(value)
#     return variable
#
#
# NickName=get_var("NickName")
# Sex=get_var("Sex")
# Province=get_var('Province')
# City=get_var('City')
# Sig=get_var('Signature')
# from pandas import DataFrame
# data={'Nickname':[NickName],'Sex':[Sex],'Province':[Province],'City':[City],'Signature':[Sig]}
# frame=DataFrame(data)
# frame.to_csv('data.csv',index=True)
#
#
# import re
# siglist=[]
# for i in friends:
#     signature=i["Signature"].strip().replace("span","").replace("class","").replace("emoji","")
#     rep=re.compile("1f\d+\w*|[<>/=]")
#     signature=rep.sub("",signature)
#     siglist.append(signature)
# text="".join(siglist)
#
# import jieba
# worldlist=jieba.cut(text,cut_all=True)
# word_space_split=" ".join(worldlist)
#
#
#
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud, ImageColorGenerator
# import PIL.Image as Image
# # coloring = np.array(Image.open("/wechat.jpg"))
# my_wordcloud = WordCloud(background_color="white", max_words=2000,
#                           max_font_size=60, random_state=42, scale=2,
#                          ).generate(word_space_split)
#
# # image_colors = ImageColorGenerator(coloring)
# # plt.imshow(my_wordcloud.recolor(color_func=image_colors))
# plt.imshow(my_wordcloud)
# plt.axis("off")
# plt.show()
#
#
#
#
import gym
import cv2
import numpy as np
# from pygame.locals import *
# import pygame,sys

# from pynput.mouse import Button,Controller
from envs import *

env = create_atari_env('MontezumaRevenge-v0')
state=env.reset()[6:10,20:60]
cv2.imshow('img',state)
cv2.waitKey()


print(np.shape(state))
