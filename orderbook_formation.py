import os
import json
import collections
import operator
from time import time as tm

if __name__ == '__main__':

    logdir = r'/home/prakhar/stock/main_data/all_files'
    os.chdir(logdir)
    file_names = os.listdir(logdir)
    basedir = r'/home/prakhar/stock'

    # List of token ids to be parsed
    id = [10794, 625]

    file_names = sorted(file_names)

    # dictionary having token as a key
    token_ds = {}
    token_strings = {}
    token_count = {}

    start_time = tm()


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


                    if token_ds[token]['sell_dict_orderid'].has_key(orderid):
                        old_qty = token_ds[token]['sell_dict_orderid'][orderid][1]
                        
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
                    
                    if token_ds[token]['buy_dict_orderid'].has_key(orderid):
                        old_qty = token_ds[token]['buy_dict_orderid'][orderid][1]


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

                        else:
                            token_ds[token]['sell_dict'][token_ds[token]['sell_dict_orderid'][sellorderid][0]][
                                'total'] -= old_qty
                            token_ds[token]['sell_dict_orderid'][sellorderid][1] = 0
                            token_ds[token]['sell_dict'][token_ds[token]['sell_dict_orderid'][sellorderid][0]][
                                sellorderid] = 0
                            
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

                        else:
                            token_ds[token]['buy_dict'][token_ds[token]['buy_dict_orderid'][buyorderid][0]][
                                'total'] -= old_qty
                            token_ds[token]['buy_dict_orderid'][buyorderid][1] = 0
                            token_ds[token]['buy_dict'][token_ds[token]['buy_dict_orderid'][buyorderid][0]][
                                buyorderid] = 0
                            
                        if token_ds[token]['buy_dict'][token_ds[token]['buy_dict_orderid'][buyorderid][0]][
                            'total'] == 0:
                            token_ds[token]['buy_price'].remove(token_ds[token]['buy_dict_orderid'][buyorderid][0])
                            token_ds[token]['buy_dict'].pop(token_ds[token]['buy_dict_orderid'][buyorderid][0])
                            token_ds[token]['buy_dict_orderid'].pop(buyorderid, None)
                        elif token_ds[token]['buy_dict_orderid'][buyorderid][1] == 0:
                            token_ds[token]['buy_dict'][token_ds[token]['buy_dict_orderid'][buyorderid][0]].pop(
                                buyorderid, None)
                            token_ds[token]['buy_dict_orderid'].pop(buyorderid, None)

                token_ds[token]['trade_data'].append([price, time])
            
            # writing orderbook  in csv file
            book_row = ''

            if len(token_ds[token]['buy_price']) > 9 and len(token_ds[token]['sell_price']) > 9 and len(
                    token_ds[token]['trade_data']) > 0:
                for i in range(0, 10):
                    book_row += str(token_ds[token]['sell_price'][i]) + ',' + str(
                        token_ds[token]['sell_dict'][token_ds[token]['sell_price'][i]]['total']) + ','
                    book_row += str(token_ds[token]['buy_price'][-i - 1]) + ',' + str(
                        token_ds[token]['buy_dict'][token_ds[token]['buy_price'][-i - 1]]['total']) + ','
                book_row += str(time)
               
                #book_row += str(token_ds[token]['trade_data'][-1][0])
                book_row += '\n'
                token_strings[token].append(book_row)

            nlines += 1

    token_count = sorted(token_count.items(), key=operator.itemgetter(1))

    file_n = r"/home/prakhar/stock/token_count.txt"
    file = open(file_n, "w")
    file.write(str(token_count))

    headings = 'ask_1,askq_1,bid_1,bidq_1,ask_2,askq_2,bid_2,bidq_2,ask_3,askq_3,bid_3,bidq_3,ask_4,askq_4,bid_4,' \
               'bidq_4,ask_5,askq_5,bid_5,bidq_5,ask_6,askq_6,bid_6,bidq_6,ask_7,askq_7,bid_7,bidq_7,ask_8,askq_8,' \
               'bid_8,bidq_8,ask_9,askq_9,bid_9,bidq_9,ask_10,askq_10,bid_10,bidq_10,timestamp'


    csv_path = r'/home/prakhar/stock/main_data/single/orderbook_'
    
    for key in token_strings.keys():
        csv_file = open(csv_path + str(key) + '.csv', 'wb')
        csv_file.write(headings)
        for row in token_strings[key]:
            csv_file.write(row)
            
    end_time = tm()
    print(end_time - start_time)
