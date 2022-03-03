import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

# Полезные функции
from utils import prefilter_items, get_recommendations


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, weighting=True):

        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать

        # применим предварительную фильтрацию к данным, убрав неинтересные для ревомендаций товары
        data = prefilter_items(data)

        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, \
        self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

        self.data = data

    @staticmethod
    def prepare_matrix(data):

        # your_code
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',  # Можно пробовать другие варианты
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)  # необходимый тип матрицы для implicit

        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=False)

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=False)

        return model

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        # your_code
        # Практически полностью реализовали на прошлом вебинаре
        # получим таблицу, содержащую по топ-N записей для каждого пользователя
        # с товарами, которые он покупал больше других в сортировке по убыванию
        user_data = self.data[self.data['user_id'] == user]
        popularity = user_data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        popularity.sort_values('quantity', ascending=False, inplace=True)

        popularity = popularity.groupby('user_id').head(N)
        popularity.sort_values(by=['user_id', 'quantity'], ascending=False, inplace=True)

        # функция возвращает id товара, похожего на переданный в параметре item_id
        def get_rec_similar_items(model, item_id):
            recs = model.similar_items(self.itemid_to_id[item_id], N=2)
            top_rec = recs[1][0]
            return self.id_to_itemid[top_rec]

        # добавим колонку с похожим товаром для каждой записи
        popularity['similar_als_recommends'] = popularity['item_id'].apply(
            lambda x: get_rec_similar_items(self.model, x))
        res = popularity['similar_als_recommends'].tolist()

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        # your_code
        users_rec_list = []
        sparse_user_item = csr_matrix(self.user_item_matrix).tocsr()
        similar_users = self.model.similar_users(self.userid_to_id[user], N=N + 1)

        for sim_user in similar_users:
            rec4user = get_recommendations(sim_user[0], self.own_recommender, sparse_user_item, N=1)
            users_rec_list.append(rec4user)

        res = [self.id_to_itemid[rec[0]] for rec in users_rec_list[1:]]

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res