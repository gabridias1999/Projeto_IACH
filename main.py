import surprise
import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader
import numpy as np
from collections import defaultdict
from operator import itemgetter
import heapq
#import recmetrics não conseguimos instalar

import os
import csv

def load_into_df():
    names = ['user', 'item', 'rating', 'appid']
    lista_jogados_user = []
    dicio_jogos_jogados_user = {}
    df = pd.read_csv("final_dataset1.csv",header=None, names=names)
    dicio_jogos = {}
    appid = list(df['appid'])
    item = list(df['item'])
    user = list(df['user'])
    lista_users = list(df['user'])
    for i in range(len(appid)):
        dicio_jogos[appid[i]] = item[i]
    for k in range(len(appid)):
        lista_jogados_user.append([user[k],item[k]])
    for l in range(len(appid)):
        if lista_jogados_user[l][0] not in dicio_jogos_jogados_user:
            lista_aux = [lista_jogados_user[l][1]]
            dicio_jogos_jogados_user[lista_jogados_user[l][0]] = lista_aux
        elif lista_jogados_user[l][0] in dicio_jogos_jogados_user:
            dicio_jogos_jogados_user[lista_jogados_user[l][0]].insert(0,lista_jogados_user[l][1])
    df = df.drop([0]) #drop na primeira linha
    df = df.drop(columns=['item']) #drop na coluna de item
    #print(df)
    #print(dicio_jogos)
    #print(dicio_jogos_jogados_user)
    #print(lista_jogados_user)
    return df,dicio_jogos, dicio_jogos_jogados_user, lista_users

def load_dataset(df):
    reader = Reader(line_format='user rating item', sep=',', skip_lines=0)
    names = ['user','appid','rating']
    df = df.reindex(columns=names)
    ratings_dataset = surprise.dataset.Dataset.load_from_df(df, reader=reader)
    return ratings_dataset

def nome_jogo(id):
    names = ['user', 'item', 'rating', 'appid']
    df = pd.read_csv("final_dataset1.csv",header=None, names=names)
    dicio_jogos = {}
    appid = list(df['appid'])
    item = list(df['item'])
    for i in range(len(appid)):
        dicio_jogos[appid[i]] = item[i]
    #print(dicio_jogos)
    if id in dicio_jogos:
        print(dicio_jogos[id])
        return dicio_jogos[id]
    else:
        print("Jogo inexistente")
        return None

def similarity_matrix(ratings_dataset):
    trainset = ratings_dataset.build_full_trainset()
    similarity_matrix = KNNBasic(sim_options={'name': 'cosine','user_based': True}).fit(trainset).compute_similarities()
    return similarity_matrix, trainset

def user_top_rated_items_e_recomendar(lista_users, trainset, dicio_jogos_jogados_user):
    user = '178964139'
    possiveis = defaultdict(float)
    counter = 0
    recomendados = []
    jogados = []
    jogados.append(dicio_jogos_jogados_user[user])
    #print(jogados, "DICIO")
    nr_jogos = 5 #nr jogos aos quais o user deu review
    if user not in lista_users:
        print("Utilizador desconhecido")
        return 0
    else:
        user_inner_id = trainset.to_inner_uid(user)
        #print("user inner id:", user_inner_id)
        user_ratings = trainset.ur[user_inner_id]
        #print("user ratings",user_ratings)
        nr_jogos_neighbors = heapq.nlargest(nr_jogos, user_ratings, key=lambda t: t[1])
        #print("neighbors",nr_jogos_neighbors)
        for id_jogo, rating in nr_jogos_neighbors:
            #print(id_jogo, "ITEM ID")
            #print(rating, "rating")
            try:
                semelhancas = similarity_matrix[id_jogo]
                for inner_id, score in enumerate(semelhancas):
                    possiveis[inner_id] += score * (rating / 5.0)
            except:
                continue
    #print("possiveis",possiveis.items())
    print("\n\nJogos recomendados:")
    for id_jogo, rating_sum in sorted(possiveis.items(), key=itemgetter(1), reverse=True):
        if id_jogo not in jogados:
            #print(id_jogo, "ITEM ID")
            #print("RATING", rating)
            #print("rating sum",rating_sum)
            #print("asdf", trainset.to_raw_iid(inner_id))
            #print(str(trainset.to_raw_iid(id_jogo)),"FDHUISFDHSFJDOSFD")
            recomendados.append(nome_jogo(str(trainset.to_raw_iid(id_jogo))))
            counter += 1
            if (counter >= 5):
                break

if __name__ == '__main__':
    df,dicio_jogos, dicio_jogos_jogados_user, lista_users = load_into_df()
    ratings_dataset = load_dataset(df)
    #print("sdffd")
    #nome_jogo("203160")
    #ratings()
    similarity_matrix, trainset = similarity_matrix(ratings_dataset)
    user_top_rated_items_e_recomendar(lista_users, trainset,dicio_jogos_jogados_user)
    #jogados_pelo_user = jogados_pelo_user()
    #jogo = nome_jogo(id)
