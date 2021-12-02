# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:27:16 2020

@author: mikf
"""

class DictToClass():
    def __init__(self,dic):
        for k,v in dic.items():
            setattr(self,k,v)
    def _get_variables(self):
        x = filter(lambda x: not x.startswith('_'),dir(self))
        return x

if __name__ == '__main__':
    dic = {'a':4,'b':345}
    Dic = DictToClass(dic)
