import os
import json
import collections
import operator
from time import time as tm

if __name__ == '__main__':

    logdir = r'/home/prakhar/stock/full_data/day_2/4'
    os.chdir(logdir)
    file_names = os.listdir(logdir)
    basedir = r'/home/prakhar/stock'
    id = [8479,4668,14356,772,2475]
    #id = [625 , 10794 , 11373]
    file_names = sorted(file_names)

    # dictionary having token as a key
    token_ds = {}
    token_strings = {}
    token_count = {}
    start_time = tm()

    total_trade_qty = 0
    total_trades = 0
    buy_miss_trade = 0
    buy_miss_qty = 0
    sell_miss_trade = 0
    sell_miss_qty = 0
    miss_total_trades = 0
    miss_total_qty = 0
    rep_new_order = 0
    orderid_miss = []

    def binarySearch(arr, l, r, x):
        if r >= l:
            mid = l + (r - l) / 2
            if (arr[mid] == x or (arr[mid - 1] < x and arr[mid] > x)):
                return mid
            elif arr[mid] > x:
                return binarySearch(arr, l, mid - 1, x)
            else:
                return binarySearch(arr, mid + 1, r, x)
        else:
            return -1


    nlines = 0
    for file_name in file_names:
        json_file = open(file_name, 'r')
        json_data = json.load(json_file)
        list_data = json_data['data']

        print(file_name)
        for item in list_data:

            ticktype = item['msg_type']

            if ticktype != 'M' and ticktype != 'N' and ticktype != 'X' and ticktype != 'T':
                continue
            token = item['token']

            if token not in id:
            #if token != id:
                continue

            if token_count.has_key(token):
                token_count[token] += 1
            else:
                token_count[token] = 1

            if not token_ds.has_key(token):
                # data structures for order book formation
                token_ds[token] = {}
                token_strings[token] = []
                token_ds[token]['buy_price'] = []
                token_ds[token]['buy_dict'] = {}
                token_ds[token]['buy_dict_orderid'] = {}
                token_ds[token]['sell_price'] = []
                token_ds[token]['sell_dict'] = {}
                token_ds[token]['sell_dict_orderid'] = {}
                token_ds[token]['trade_data'] = []

                # time sensitive variables
                token_ds[token]['rate_new_buy'] = collections.deque()
                token_ds[token]['new_qty_buy'] = 0
                token_ds[token]['new_price_buy'] = 0
                token_ds[token]['rate_new_sell'] = collections.deque()
                token_ds[token]['new_qty_sell'] = 0
                token_ds[token]['new_price_sell'] = 0
                token_ds[token]['rate_cancel_buy'] = collections.deque()
                token_ds[token]['cancel_qty_buy'] = 0
                token_ds[token]['cancel_price_buy'] = 0
                token_ds[token]['rate_cancel_sell'] = collections.deque()
                token_ds[token]['cancel_qty_sell'] = 0
                token_ds[token]['cancel_price_sell'] = 0
                token_ds[token]['rate_trade'] = collections.deque()
                token_ds[token]['trade_sum_qty'] = 0
                token_ds[token]['trade_sum_price'] = 0
                token_ds[token]['trade_qty_price'] = 0
                token_ds[token]['trade_qty_pos'] = 0
                token_ds[token]['buy_sell_rec'] = collections.deque()

                # data structures for tracking substantial buyers/sellers
                # token_ds[token]['distinct_order_ids_sell'] = {}
                # token_ds[token]['distinct_order_ids_buy'] = {}
                # token_ds[token]['distinct_order_ids_sell']['total_qty'] = 0
                # token_ds[token]['distinct_order_ids_sell']['avg_price'] = 0
                # token_ds[token]['distinct_order_ids_buy']['total_qty'] = 0
                # token_ds[token]['distinct_order_ids_buy']['avg_price'] = 0

            # print item
            price = item['price']
            qty = item['quantity']
            time = item['timestamp']
            time_ms = time / 1000
            index = 0

            # print(nlines)

            if ticktype == 'N':
                signal = item['order_type']
                orderid = item['order_id']
                if signal == 'S':
                    if token_ds[token]['sell_dict_orderid'].has_key(orderid):
                        rep_new_order += 1
                    # for tracking substantial sellers/buyers
                    # token_ds[token]['distinct_order_ids_sell'][orderid] = 0
                    # token_ds[token]['distinct_order_ids_sell']['total_qty'] += qty
                    # token_ds[token]['distinct_order_ids_sell']['avg_price'] += price

                    # for calculating rate of change of inflow/outflow of orders in market
                    token_ds[token]['rate_new_sell'].append([time_ms, price, qty])
                    token_ds[token]['new_qty_sell'] += qty
                    token_ds[token]['new_price_sell'] += price

                    while token_ds[token]['rate_new_sell'][-1][0] - token_ds[token]['rate_new_sell'][0][0] > 60:
                        token_ds[token]['new_qty_sell'] -= token_ds[token]['rate_new_sell'][0][2]
                        token_ds[token]['new_price_sell'] -= token_ds[token]['rate_new_sell'][0][1]
                        token_ds[token]['rate_new_sell'].popleft()

                    # for formation of orderbook
                    if not token_ds[token]['sell_dict'].has_key(price):
                        if len(token_ds[token]['sell_price']) == 0:
                            token_ds[token]['sell_price'].append(price)
                        elif price < token_ds[token]['sell_price'][0]:
                            token_ds[token]['sell_price'].insert(0, price)
                        elif price > token_ds[token]['sell_price'][-1]:
                            token_ds[token]['sell_price'].append(price)
                        else:
                            index = binarySearch(token_ds[token]['sell_price'], 0,
                                                 len(token_ds[token]['sell_price']) - 1, price)
                            token_ds[token]['sell_price'].insert(index, price)
                        token_ds[token]['sell_dict'][price] = {}
                        token_ds[token]['sell_dict'][price]['total'] = 0
                        # print 100
                    token_ds[token]['sell_dict'][price][orderid] = qty
                    token_ds[token]['sell_dict'][price]['total'] += qty
                    token_ds[token]['sell_dict_orderid'][orderid] = [price, qty]

                else:
                    # for tracking substantial sellers/buyers
                    # token_ds[token]['distinct_order_ids_buy'][orderid] = 0
                    # token_ds[token]['distinct_order_ids_buy']['total_qty'] += qty
                    # token_ds[token]['distinct_order_ids_buy']['avg_price'] += price

                    # for calculating rate of change of inflow/outflow of orders in market
                    token_ds[token]['rate_new_buy'].append([time, price, qty])
                    token_ds[token]['new_qty_buy'] += qty
                    token_ds[token]['new_price_buy'] += price

                    while token_ds[token]['rate_new_buy'][-1][0] - token_ds[token]['rate_new_buy'][0][0] > 60:
                        token_ds[token]['new_qty_buy'] -= token_ds[token]['rate_new_buy'][0][2]
                        token_ds[token]['new_price_buy'] -= token_ds[token]['rate_new_buy'][0][1]
                        token_ds[token]['rate_new_buy'].popleft()

                    # for formation of orderbook
                    if not token_ds[token]['buy_dict'].has_key(price):
                        if len(token_ds[token]['buy_price']) == 0:
                            token_ds[token]['buy_price'].append(price)
                        elif price < token_ds[token]['buy_price'][0]:
                            token_ds[token]['buy_price'].insert(0, price)
                        elif price > token_ds[token]['buy_price'][-1]:
                            token_ds[token]['buy_price'].append(price)
                        else:
                            index = binarySearch(token_ds[token]['buy_price'], 0, len(token_ds[token]['buy_price']) - 1,
                                                 price)
                            token_ds[token]['buy_price'].insert(index, price)
                        token_ds[token]['buy_dict'][price] = {}
                        token_ds[token]['buy_dict'][price]['total'] = 0
                        # print 10
                    token_ds[token]['buy_dict'][price][orderid] = qty
                    token_ds[token]['buy_dict'][price]['total'] += qty
                    token_ds[token]['buy_dict_orderid'][orderid] = [price, qty]

            elif ticktype == 'M':
                signal = item['order_type']
                orderid = item['order_id']
                if signal == 'S':

                    if token_ds[token]['sell_dict_orderid'].has_key(orderid):
                        old_price = token_ds[token]['sell_dict_orderid'][orderid][0]
                        old_qty = token_ds[token]['sell_dict_orderid'][orderid][1]

                        # for tracking substantial sellers/buyers
                        # if token_ds[token]['distinct_order_ids_sell'].has_key(orderid):
                        #    token_ds[token]['distinct_order_ids_sell'][orderid] += 1
                        #    if (token_ds[token]['distinct_order_ids_sell'][orderid] > 3):
                        #        token_ds[token]['distinct_order_ids_sell'].pop(orderid, None)
                        #        token_ds[token]['distinct_order_ids_sell']['total_qty'] -= old_qty
                        #        token_ds[token]['distinct_order_ids_sell']['avg_price'] -= old_price
                        #    else:
                        #        token_ds[token]['distinct_order_ids_sell']['total_qty'] += qty - old_qty
                        #        token_ds[token]['distinct_order_ids_sell']['avg_price'] += price - old_price
                    # else:
                    # for tracking substantial sellers/buyers
                    # token_ds[token]['distinct_order_ids_sell'][orderid] = 0
                    # token_ds[token]['distinct_order_ids_sell']['total_qty'] += qty
                    # token_ds[token]['distinct_order_ids_sell']['avg_price'] += price

                    # for formation of orderbook
                    if not token_ds[token]['sell_dict'].has_key(price):
                        if price < token_ds[token]['sell_price'][0]:
                            token_ds[token]['sell_price'].insert(0, price)
                        elif price > token_ds[token]['sell_price'][-1]:
                            token_ds[token]['sell_price'].append(price)
                        else:
                            index = binarySearch(token_ds[token]['sell_price'], 0,
                                                 len(token_ds[token]['sell_price']) - 1, price)
                            token_ds[token]['sell_price'].insert(index, price)
                        token_ds[token]['sell_dict'][price] = {}
                        token_ds[token]['sell_dict'][price]['total'] = 0
                    if token_ds[token]['sell_dict_orderid'].has_key(orderid):
                        if price != old_price:
                            token_ds[token]['sell_dict'][price][orderid] = qty
                            token_ds[token]['sell_dict'][price]['total'] += qty

                            token_ds[token]['sell_dict'][old_price]['total'] -= old_qty
                            token_ds[token]['sell_dict'][old_price].pop(orderid, None)
                            if token_ds[token]['sell_dict'][old_price]['total'] == 0:
                                token_ds[token]['sell_price'].remove(old_price)
                                token_ds[token]['sell_dict'].pop(old_price, None)
                        else:
                            token_ds[token]['sell_dict'][price][orderid] = qty
                            token_ds[token]['sell_dict'][price]['total'] = token_ds[token]['sell_dict'][price][
                                                                               'total'] + qty - old_qty
                    else:
                        token_ds[token]['sell_dict'][price][orderid] = qty
                        token_ds[token]['sell_dict'][price]['total'] += qty

                    token_ds[token]['sell_dict_orderid'][orderid] = [price, qty]


                else:
                    if token_ds[token]['buy_dict_orderid'].has_key(orderid):
                        old_price = token_ds[token]['buy_dict_orderid'][orderid][0]
                        old_qty = token_ds[token]['buy_dict_orderid'][orderid][1]

                        # for tracking substantial sellers/buyers
                        # if token_ds[token]['distinct_order_ids_buy'].has_key(orderid):
                        #    token_ds[token]['distinct_order_ids_buy'][orderid] += 1
                        #    if token_ds[token]['distinct_order_ids_buy'][orderid] > 3:
                        #        token_ds[token]['distinct_order_ids_buy'].pop(orderid, None)
                        #        token_ds[token]['distinct_order_ids_buy']['total_qty'] -= old_qty
                        #        token_ds[token]['distinct_order_ids_buy']['avg_price'] -= old_price
                        #    else:
                        #        token_ds[token]['distinct_order_ids_buy']['total_qty'] += qty - old_qty
                        #        token_ds[token]['distinct_order_ids_buy']['avg_price'] += price - old_price
                    # else:
                    # token_ds[token]['distinct_order_ids_buy'][orderid] = 0
                    # token_ds[token]['distinct_order_ids_buy']['total_qty'] += qty
                    # token_ds[token]['distinct_order_ids_buy']['avg_price'] += price

                    # for formation of orderbook
                    if not token_ds[token]['buy_dict'].has_key(price):
                        if price < token_ds[token]['buy_price'][0]:
                            token_ds[token]['buy_price'].insert(0, price)
                        elif price > token_ds[token]['buy_price'][-1]:
                            token_ds[token]['buy_price'].append(price)
                        else:
                            index = binarySearch(token_ds[token]['buy_price'], 0, len(token_ds[token]['buy_price']) - 1,
                                                 price)
                            token_ds[token]['buy_price'].insert(index, price)
                        token_ds[token]['buy_dict'][price] = {}
                        token_ds[token]['buy_dict'][price]['total'] = 0
                    if token_ds[token]['buy_dict_orderid'].has_key(orderid):
                        if price != old_price:
                            token_ds[token]['buy_dict'][price][orderid] = qty
                            token_ds[token]['buy_dict'][price]['total'] += qty

                            token_ds[token]['buy_dict'][old_price]['total'] -= old_qty
                            token_ds[token]['buy_dict'][old_price].pop(orderid, None)
                            if token_ds[token]['buy_dict'][old_price]['total'] == 0:
                                token_ds[token]['buy_price'].remove(old_price)
                                token_ds[token]['buy_dict'].pop(old_price, None)
                        else:
                            token_ds[token]['buy_dict'][price][orderid] = qty
                            token_ds[token]['buy_dict'][price]['total'] = token_ds[token]['buy_dict'][price][
                                                                              'total'] + qty - old_qty
                    else:
                        token_ds[token]['buy_dict'][price][orderid] = qty
                        token_ds[token]['buy_dict'][price]['total'] += qty

                    token_ds[token]['buy_dict_orderid'][orderid] = [price, qty]


            elif ticktype == 'X':
                orderid = item['order_id']
                signal = item['order_type']
                if signal == 'S':

                    # for calculating rate of change of inflow/outflow of orders in market
                    token_ds[token]['rate_cancel_sell'].append([time_ms, price, qty])
                    token_ds[token]['cancel_qty_sell'] += qty
                    token_ds[token]['cancel_price_sell'] += price

                    while token_ds[token]['rate_cancel_sell'][-1][0] - token_ds[token]['rate_cancel_sell'][0][0] > 60:
                        token_ds[token]['cancel_qty_sell'] -= token_ds[token]['rate_cancel_sell'][0][2]
                        token_ds[token]['cancel_price_sell'] -= token_ds[token]['rate_cancel_sell'][0][1]
                        token_ds[token]['rate_cancel_sell'].popleft()

                    if token_ds[token]['sell_dict_orderid'].has_key(orderid):
                        old_qty = token_ds[token]['sell_dict_orderid'][orderid][1]
                        # if token_ds[token]['distinct_order_ids_sell'].has_key(orderid):
                        #    token_ds[token]['distinct_order_ids_sell'].pop(orderid, None)
                        #    token_ds[token]['distinct_order_ids_sell']['total_qty'] -= old_qty
                        #    token_ds[token]['distinct_order_ids_sell']['avg_price'] -= \
                        #    token_ds[token]['sell_dict_orderid'][orderid][0]

                        # for formation of orderbook
                        token_ds[token]['sell_dict'][token_ds[token]['sell_dict_orderid'][orderid][0]][
                            'total'] -= old_qty
                        token_ds[token]['sell_dict_orderid'][orderid][1] = 0
                        if token_ds[token]['sell_dict'][token_ds[token]['sell_dict_orderid'][orderid][0]]['total'] == 0:
                            token_ds[token]['sell_price'].remove(token_ds[token]['sell_dict_orderid'][orderid][0])
                            token_ds[token]['sell_dict'].pop(token_ds[token]['sell_dict_orderid'][orderid][0])
                        else:
                            token_ds[token]['sell_dict'][token_ds[token]['sell_dict_orderid'][orderid][0]].pop(orderid,
                                                                                                               None)
                        token_ds[token]['sell_dict_orderid'].pop(orderid, None)

                else:
                    # for calculating rate of change of inflow/outflow of orders in market
                    token_ds[token]['rate_cancel_buy'].append([time_ms, price, qty])
                    token_ds[token]['cancel_qty_buy'] += qty
                    token_ds[token]['cancel_price_buy'] += price
                    while token_ds[token]['rate_cancel_buy'][-1][0] - token_ds[token]['rate_cancel_buy'][0][0] > 60:
                        token_ds[token]['cancel_qty_buy'] -= token_ds[token]['rate_cancel_buy'][0][2]
                        token_ds[token]['cancel_price_buy'] -= token_ds[token]['rate_cancel_buy'][0][1]
                        token_ds[token]['rate_cancel_buy'].popleft()

                    if token_ds[token]['buy_dict_orderid'].has_key(orderid):
                        old_qty = token_ds[token]['buy_dict_orderid'][orderid][1]

                        # for tracking substantial buyers/sellers
                        # if token_ds[token]['distinct_order_ids_buy'].has_key(orderid):
                        #     token_ds[token]['distinct_order_ids_buy'].pop(orderid, None)
                        #     token_ds[token]['distinct_order_ids_buy']['total_qty'] -= old_qty
                        #     token_ds[token]['distinct_order_ids_buy']['avg_price'] -= \
                        #     token_ds[token]['buy_dict_orderid'][orderid][0]

                        # for formation of orderbook
                        token_ds[token]['buy_dict'][token_ds[token]['buy_dict_orderid'][orderid][0]]['total'] -= old_qty
                        if token_ds[token]['buy_dict'][token_ds[token]['buy_dict_orderid'][orderid][0]]['total'] == 0:
                            token_ds[token]['buy_price'].remove(token_ds[token]['buy_dict_orderid'][orderid][0])
                            token_ds[token]['buy_dict'].pop(token_ds[token]['buy_dict_orderid'][orderid][0])
                        else:
                            token_ds[token]['buy_dict'][token_ds[token]['buy_dict_orderid'][orderid][0]].pop(orderid,
                                                                                                             None)
                        token_ds[token]['buy_dict_orderid'].pop(orderid, None)

            elif ticktype == 'T':
                buyorderid = item['buy_order_id']
                sellorderid = item['sell_order_id']

                total_trade_qty += qty
                total_trades += 1

                if ((not token_ds[token]['buy_dict_orderid'].has_key(buyorderid)) and buyorderid != '0.000000') or \
                        ((not token_ds[token]['sell_dict_orderid'].has_key(sellorderid)) and sellorderid !='0.000000'):
                    miss_total_trades += 1
                    miss_total_qty += qty


                # for calculating rate of change of inflow/outflow of orders in market
                if len(token_ds[token]['buy_price']) > 0 and len(token_ds[token]['sell_price']) > 0:
                    token_ds[token]['trade_sum_price'] += price
                    if abs(price - token_ds[token]['sell_price'][0]) > abs(price - token_ds[token]['buy_price'][-1]):
                        token_ds[token]['rate_trade'].append([time_ms, price, qty])
                        token_ds[token]['trade_sum_qty'] += qty
                    else:
                        token_ds[token]['rate_trade'].append([time_ms, price, -qty])
                        token_ds[token]['trade_sum_qty'] += -qty

                    token_ds[token]['trade_qty_pos'] += qty
                    token_ds[token]['trade_qty_price'] += price * qty

                    while token_ds[token]['rate_trade'][-1][0] - token_ds[token]['rate_trade'][0][0] > 60:
                        token_ds[token]['trade_sum_qty'] -= token_ds[token]['rate_trade'][0][2]
                        token_ds[token]['trade_sum_price'] -= token_ds[token]['rate_trade'][0][1]
                        token_ds[token]['trade_qty_pos'] -= abs(token_ds[token]['rate_trade'][0][2])
                        token_ds[token]['trade_qty_price'] -= token_ds[token]['rate_trade'][0][1] * abs(
                            token_ds[token]['rate_trade'][0][2])
                        token_ds[token]['rate_trade'].popleft()

                # for formation of orderbook
                if sellorderid != '0.000000':
                    if token_ds[token]['sell_dict_orderid'].has_key(sellorderid):

                        old_qty = token_ds[token]['sell_dict_orderid'][sellorderid][1]
                        if (old_qty - qty) >= 0:
                            token_ds[token]['sell_dict'][token_ds[token]['sell_dict_orderid'][sellorderid][0]][
                                'total'] -= qty
                            token_ds[token]['sell_dict_orderid'][sellorderid][1] -= qty
                            token_ds[token]['sell_dict'][token_ds[token]['sell_dict_orderid'][sellorderid][0]][
                                sellorderid] -= qty
                            # if token_ds[token]['distinct_order_ids_sell'].has_key(sellorderid):
                            #     token_ds[token]['distinct_order_ids_sell']['total_qty'] -= qty

                        else:
                            token_ds[token]['sell_dict'][token_ds[token]['sell_dict_orderid'][sellorderid][0]][
                                'total'] -= old_qty
                            token_ds[token]['sell_dict_orderid'][sellorderid][1] = 0
                            token_ds[token]['sell_dict'][token_ds[token]['sell_dict_orderid'][sellorderid][0]][
                                sellorderid] = 0
                            #     if token_ds[token]['distinct_order_ids_sell'].has_key(sellorderid):
                            #         token_ds[token]['distinct_order_ids_sell']['total_qty'] -= old_qty
                            #
                            # if token_ds[token]['distinct_order_ids_sell'].has_key(sellorderid):
                            #     if token_ds[token]['sell_dict_orderid'][sellorderid][1] == 0:
                            token_ds[token]['distinct_order_ids_sell'].pop(sellorderid, None)
                            token_ds[token]['distinct_order_ids_sell']['avg_price'] -= \
                                token_ds[token]['sell_dict_orderid'][sellorderid][0]

                        if token_ds[token]['sell_dict'][token_ds[token]['sell_dict_orderid'][sellorderid][0]][
                            'total'] == 0:
                            token_ds[token]['sell_price'].remove(token_ds[token]['sell_dict_orderid'][sellorderid][0])
                            token_ds[token]['sell_dict'].pop(token_ds[token]['sell_dict_orderid'][sellorderid][0])
                            token_ds[token]['sell_dict_orderid'].pop(sellorderid, None)
                        elif token_ds[token]['sell_dict_orderid'][sellorderid][1] == 0:
                            token_ds[token]['sell_dict'][token_ds[token]['sell_dict_orderid'][sellorderid][0]].pop(
                                sellorderid, None)
                            token_ds[token]['sell_dict_orderid'].pop(sellorderid, None)
                    else:
                        sell_miss_qty += qty
                        sell_miss_trade += 1
                        orderid_miss.append(sellorderid)
                # for formation of orderbook
                if buyorderid != '0.000000':
                    if token_ds[token]['buy_dict_orderid'].has_key(buyorderid):

                        old_qty = token_ds[token]['buy_dict_orderid'][buyorderid][1]
                        if old_qty - qty >= 0:
                            token_ds[token]['buy_dict'][token_ds[token]['buy_dict_orderid'][buyorderid][0]][
                                'total'] -= qty
                            token_ds[token]['buy_dict_orderid'][buyorderid][1] -= qty
                            token_ds[token]['buy_dict'][token_ds[token]['buy_dict_orderid'][buyorderid][0]][
                                buyorderid] -= qty
                            # if token_ds[token]['distinct_order_ids_buy'].has_key(buyorderid):
                            #     token_ds[token]['distinct_order_ids_buy']['total_qty'] -= qty

                        else:
                            token_ds[token]['buy_dict'][token_ds[token]['buy_dict_orderid'][buyorderid][0]][
                                'total'] -= old_qty
                            token_ds[token]['buy_dict_orderid'][buyorderid][1] = 0
                            token_ds[token]['buy_dict'][token_ds[token]['buy_dict_orderid'][buyorderid][0]][
                                buyorderid] = 0
                            #     if token_ds[token]['distinct_order_ids_buy'].has_key(buyorderid):
                            #         token_ds[token]['distinct_order_ids_buy']['total_qty'] -= old_qty
                            # if token_ds[token]['distinct_order_ids_buy'].has_key(buyorderid):
                            #     if token_ds[token]['buy_dict_orderid'][buyorderid][1] == 0:
                            #         token_ds[token]['distinct_order_ids_buy'].pop(buyorderid, None)
                            # token_ds[token]['distinct_order_ids_buy']['avg_price'] -= \
                            # token_ds[token]['buy_dict_orderid'][buyorderid][0]

                        if token_ds[token]['buy_dict'][token_ds[token]['buy_dict_orderid'][buyorderid][0]][
                            'total'] == 0:
                            token_ds[token]['buy_price'].remove(token_ds[token]['buy_dict_orderid'][buyorderid][0])
                            token_ds[token]['buy_dict'].pop(token_ds[token]['buy_dict_orderid'][buyorderid][0])
                            token_ds[token]['buy_dict_orderid'].pop(buyorderid, None)
                        elif token_ds[token]['buy_dict_orderid'][buyorderid][1] == 0:
                            token_ds[token]['buy_dict'][token_ds[token]['buy_dict_orderid'][buyorderid][0]].pop(
                                buyorderid, None)
                            token_ds[token]['buy_dict_orderid'].pop(buyorderid, None)
                    else:
                        buy_miss_qty += qty
                        buy_miss_trade += 1
                        orderid_miss.append(buyorderid)




                token_ds[token]['trade_data'].append([price, time])
            total_ask_price = 0
            total_buy_price = 0
            total_ask_qty = 0
            total_buy_qty = 0

            if len(token_ds[token]['buy_price']) > 9 and len(token_ds[token]['sell_price']) > 9:
                for i in range(0, 10):
                    total_ask_price += token_ds[token]['sell_price'][i]
                    total_buy_price += token_ds[token]['buy_price'][-i - 1]
                    total_ask_qty += token_ds[token]['sell_dict'][token_ds[token]['sell_price'][i]]['total']
                    total_buy_qty += token_ds[token]['buy_dict'][token_ds[token]['buy_price'][i]]['total']
                token_ds[token]['buy_sell_rec'].append(
                    [(token_ds[token]['sell_price'][0] + token_ds[token]['buy_price'][-1]) / 2
                        , total_ask_price / 10,
                     total_buy_price / 10, total_ask_qty / 10, total_buy_qty / 10,
                     time_ms])
                while token_ds[token]['buy_sell_rec'][-1][-1] - token_ds[token]['buy_sell_rec'][0][-1] > 60:
                    token_ds[token]['buy_sell_rec'].popleft()

            # writing orderbook  in csv file
            book_row = ''

            if len(token_ds[token]['buy_price']) > 9 and len(token_ds[token]['sell_price']) > 9 and len(
                    token_ds[token]['trade_data']) > 0:
                for i in range(0, 10):
                    book_row += str(token_ds[token]['sell_price'][i]) + ',' + str(
                        token_ds[token]['sell_dict'][token_ds[token]['sell_price'][i]]['total']) + ','
                    book_row += str(token_ds[token]['buy_price'][-i - 1]) + ',' + str(
                        token_ds[token]['buy_dict'][token_ds[token]['buy_price'][-i - 1]]['total']) + ','

                book_row += str(time) + ',' + str(len(token_ds[token]['rate_new_sell'])) + ',' + str(
                    token_ds[token]['new_price_sell']) + ',' + str(token_ds[token]['new_qty_sell']) + ',' + str(
                    len(token_ds[token]['rate_new_buy'])) + ',' + str(token_ds[token]['new_price_buy']) + ',' + str(
                    token_ds[token]['new_qty_buy'])
                book_row += ',' + str(len(token_ds[token]['rate_cancel_sell'])) + ',' + str(
                    token_ds[token]['cancel_price_sell']) + ',' + str(token_ds[token]['cancel_qty_sell']) + ',' + str(
                    len(token_ds[token]['rate_cancel_buy'])) + ',' + str(
                    token_ds[token]['cancel_price_buy']) + ',' + str(
                    token_ds[token]['cancel_qty_buy'])
                book_row += ',' + str(len(token_ds[token]['rate_trade'])) + ',' + str(
                    token_ds[token]['trade_sum_price']) + ',' + str(token_ds[token]['trade_sum_qty']) + ',' + str(
                    token_ds[token]['trade_qty_pos']) + ',' + str(token_ds[token]['trade_qty_price']) + ','

                for i in range(0, 5):
                    book_row += str(token_ds[token]['buy_sell_rec'][-1][i]) + ','
                    book_row += str(token_ds[token]['buy_sell_rec'][0][i]) + ','

                # book_row += str(len(token_ds[token]['distinct_order_ids_sell'])) + ','
                # book_row += str(token_ds[token]['distinct_order_ids_sell']['total_qty']) + ','
                # book_row += str(token_ds[token]['distinct_order_ids_sell']['avg_price']) + ','
                # book_row += str(len(token_ds[token]['distinct_order_ids_buy'])) + ','
                # book_row += str(token_ds[token]['distinct_order_ids_buy']['total_qty']) + ','
                # book_row += str(token_ds[token]['distinct_order_ids_buy']['avg_price']) + ','

                book_row += str(token_ds[token]['trade_data'][-1][0])
                book_row += '\n'
                # csv_file.write(book_row)
                token_strings[token].append(book_row)

            nlines += 1

    token_count = sorted(token_count.items(), key=operator.itemgetter(1))
    """
    file_n = r"/home/prakhar/stock/token_count.txt"
    file = open(file_n, "w")
    file.write(str(token_count))
    file.write('\n' +'total_trade_qty : ' + str(total_trade_qty) +' total_trades: ' +str(total_trades) +
    ' buy_miss_trade ' + str(buy_miss_trade) + '\n' +
    ' buy_miss_qty ' + str(buy_miss_qty) + '\n' +
    'sell_miss_trade'  + str(sell_miss_trade) + '\n' +
    'sell_miss_qty'  + str(sell_miss_qty) + '\n' +
    'miss_total_trades' + str(miss_total_trades) + '\n' +
    'miss_total_qty' + str(miss_total_qty) + '\n'+'rep_new_order' + str(rep_new_order) +'\n' )
    """

    headings = 'ask_1,askq_1,bid_1,bidq_1,ask_2,askq_2,bid_2,bidq_2,ask_3,askq_3,bid_3,bidq_3,ask_4,askq_4,bid_4,' \
               'bidq_4,ask_5,askq_5,bid_5,bidq_5,ask_6,askq_6,bid_6,bidq_6,ask_7,askq_7,bid_7,bidq_7,ask_8,askq_8,' \
               'bid_8,bidq_8,ask_9,askq_9,bid_9,bidq_9,ask_10,askq_10,bid_10,bidq_10,timestamp,New_Sell_Orders_No,' \
               'Sum_Sell_Order_Prices,Total_Qty_Sell_Orders,New_Buy_Orders_No,Sum_Buy_Order_Prices,' \
               'Total_Qty_Buy_Orders,Sell_Orders_No_Cancelled,Sum_Sell_Order_Prices_Cancelled,' \
               'Total_Qty_Sell_Orders_Cancelled,Buy_Orders_No_Cancelled,Sum_Buy_Order_Prices_Cancelled,' \
               'Total_Qty_Buy_Orders_Cancelled,No_of_Trades,Trade_Price_Sum,Trade_Qty_net,trade_qty_sum,trade_vwap,' \
               'mid_price_now,mid_price_old,avg_ask_price_now,avg_ask_price_old,avg_bid_price_now,' \
               'avg_bid_price_old,avg_ask_vol_now,avg_ask_vol_old,avg_bid_vol_now,avg_bid_vol_old,LTP\n'
               


    csv_path = r'/home/prakhar/stock/full_data/csv/day_2/orderbook_'
    for key in token_strings.keys():
        csv_file = open(csv_path + str(key) + '.csv', 'wb')
        csv_file.write(headings)
        for row in token_strings[key]:
            csv_file.write(row)

    """
    file_orderid = r'/home/prakhar/stock/orderid_miss.json'
    json_file = open(file_orderid , 'w')
    json.dumps(orderid_miss , json_file)

    """

    end_time = tm()
    print(end_time - start_time)
