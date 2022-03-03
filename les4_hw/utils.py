import pandas as pd
import numpy as np


def prefilter_items(data, item_features=None):
    ## Уберем не интересные для рекоммендаций категории (department)
    # не интересные определим по количеству товаров в одной категории
    n_interest_items_treshold = 150
    if item_features is not None:
        category_size = pd.DataFrame(item_features. \
                                       groupby('department')['item_id'].nunique(). \
                                       sort_values(ascending=False)).reset_index()

        category_size.columns = ['category', 'n_items']
        not_interesting_cats = category_size[category_size['n_items'] < n_interest_items_treshold].category.tolist()
        items_in_not_interesting_cats = item_features[
            item_features['department'].isin(not_interesting_cats)].item_id.unique().tolist()

        data = data[~data['item_id'].isin(items_in_not_interesting_cats)]

    ## Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    # самые дешевые определим по стоимости не выше порга дешевизны 
    cheap_treshold = 2
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data[data['price'] > cheap_treshold]

    ## Уберем слишком дорогие товарыs
    # самые лорогие определим как превышающие порог дороговизны
    expensive_treshold = 60
    data = data[data['price'] < expensive_treshold]

    ## Уберем товары, которые не продавались за последние 12 месяцев
    # кол-во недель в 12 месяцах 52. 
    no_action_weeks = 52
    # Последние 12 месяцев 
    no_sells = data.loc[(data['week_no'] < data['week_no'].max() - no_action_weeks) & (data['quantity'] == 0)]\
                        .item_id.tolist()
    data = data[~data['item_id'].isin(no_sells)]

    ## Уберем самые популярные товары (их и так купят)
    # самые популярные определим как отношение количества покупок уникальными пользователями 
    # определенного товара к количеству уникальных пользователей. Если какой то товар купили более
    # половины пользователей, такой товар популярный
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index()
    popularity['share_unique_users'] = popularity['user_id'] / data['user_id'].nunique()
    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    ## Уберем самые НЕ популярные товары (их и так НЕ купят)
    # Если товар купили менее одного процента пользователей - товар не популярный
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]

    return data

def postfilter_items(user_id, recommednations):
    pass

def get_recommendations(user, model, sparse_user_item, N=5):
    """Рекомендуем топ-N товаров"""
    
    res = [rec[0] for rec in 
                    model.recommend(userid=user, 
                                    user_items=sparse_user_item,   # на вход user-item matrix
                                    N=N, 
                                    filter_already_liked_items=False, 
                                    filter_items=None,  # !!! 
                                    recalculate_user=True)]
    return res


