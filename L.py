# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 18:03:52 2023

@author: foolinsky
"""

# calculate l^prime in eq.6 
def lprime(l):
    a,b,c,d=1,1,1,0
    while a<=l:
        b=1
        while a*b<=l:
            c=1
            while a*b*c<=l:
                if d<a*b*c:
                    d=a*b*c
                c=c*5
            b=b*3
        a=a*2
    return d