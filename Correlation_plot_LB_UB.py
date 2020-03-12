# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 10:57:08 2019

@author: weiyong
"""

LB=[
7.75,
10.58,
8.67,
10.67,
7.92,
10.58,
8.33,
8.92,
8.00,
10.50,
10.50,
6.42,
8.67,
9.33,
11.33,
7.75,
8.50,
9.33,
6.75,
8.00,
9.08,
9.42,
6.50,
5.83,
8.33,
7.00,
8.50,
5.75,
11.33,
9.83
]

UB=[
8.67,
9.58,
10.25,
10.50,
5.42,
11.50,
8.58,
10.67,
10.75,
8.92,
10.50,
8.17,
7.25,
5.83,
11.33,
8.83,
8.67,
7.75,
6.33,
5.25,
10.33,
9.67,
8.50,
2.58,
6.92,
7.83,
6.17,
8.58,
10.25,
11.10
]

Digits_SPAN=[
27.00,
42.00,
28.00,
33.00,
33.00,
32.00,
28.00,
29.00,
32.00,
30.00,
28.00,
30.00,
27.00,
26.00,
38.00,
30.00,
31.00,
30.00,
36.00,
26.00,
22.00,
31.00,
26.00,
29.00,
30.00,
31.00,
34.00,
28.00,
44.00,
30.00
]

RAN_LETTER=[
13.00,
14.00,
14.00,
10.00,
14.00,
13.00,
20.00,
14.00,
16.00,
21.00,
19.00,
17.00,
20.00,
19.00,
15.00,
16.00,
21.00,
23.00,
24.00,
18.00,
17.00,
15.00,
27.00,
15.00,
17.00,
29.00,
17.00,
25.00,
14.00,
21.00
]

RAN_OBJECT=[
27.00,
24.00,
27.00,
29.00,
33.00,
23.00,
36.00,
31.00,
29.00,
33.00,
26.00,
31.00,
38.00,
35.00,
30.00,
26.00,
33.00,
35.00,
37.00,
40.00,
31.00,
29.00,
35.00,
39.00,
32.00,
35.00,
37.00,
30.00,
27.00,
34.00
]

PP=[
51.00,
53.00,
51.00,
51.00,
52.00,
53.00,
51.00,
52.00,
53.00,
50.00,
51.00,
51.00,
52.00,
49.00,
52.00,
53.00,
52.00,
52.00,
52.00,
51.00,
52.00,
53.00,
52.00,
50.00,
52.00,
53.00,
52.00,
52.00,
53.00,
53.00
]

import pandas as pd
import numpy as np
df=pd.DataFrame(np.array([LB,UB,Digits_SPAN,RAN_LETTER,RAN_OBJECT,PP])).transpose()
df.columns=['Learning Speed LB','Learning Speed UB','Digits Span','RAN Letters','RAN Objects','Phonological Processing']

from scipy.stats import linregress
b1, i1, r1, p1, std1 = linregress(df['Learning Speed UB'],df['Digits Span'])
b2, i2, r2, p2, std2 = linregress(df['Learning Speed UB'],df['RAN Letters'])
b3, i3, r3, p3, std3 = linregress(df['Learning Speed UB'],df['RAN Objects'])
b4, i4, r4, p4, std4 = linregress(df['Learning Speed UB'],df['Phonological Processing'])

from scipy.stats import linregress
b5, i5, r5, p5, std5 = linregress(df['Learning Speed LB'],df['Digits Span'])
b6, i6, r6, p6, std6 = linregress(df['Learning Speed LB'],df['RAN Letters'])
b7, i7, r7, p7, std7 = linregress(df['Learning Speed LB'],df['RAN Objects'])
b8, i8, r8, p8, std8 = linregress(df['Learning Speed LB'],df['Phonological Processing'])


import matplotlib.pyplot as plt
f = plt.figure()
f.set_size_inches((16, 8))
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.6)

ax = plt.subplot(241)

sns.regplot('Learning Speed UB','Digits Span',df)
text = f"$r={round(r1, 4)}$\n$p={round(p1, 4)}$"
plt.text(4, 40, text,fontsize=16)
plt.xlabel('Learning Speed')
#plt.xlim([2,12])
plt.tight_layout()

ax = plt.subplot(242)
sns.regplot('Learning Speed UB','RAN Letters',df)
text = f"$r={round(r2, 4)}$\n$p={round(p2, 4)}$"
plt.text(2.5, 25, text,fontsize=16)
plt.xlabel('Learning Speed')
plt.tight_layout()

ax = plt.subplot(243)
sns.regplot('Learning Speed UB','RAN Objects',df)
text = f"$r={round(r3, 4)}$\n$p={round(p3, 5)}$"
plt.text(6, 40, text,fontsize=16,color='red')
plt.xlabel('Learning Speed')
plt.tight_layout()

ax = plt.subplot(244)
sns.regplot('Learning Speed UB','Phonological Processing',df)
text = f"$r={round(r4, 4)}$\n$p={round(p4, 4)}$"
plt.text(3, 52.5, text,fontsize=16,color='red')
plt.xlabel('Learning Speed')
plt.tight_layout()


ax = plt.subplot(245)

sns.regplot('Learning Speed LB','Digits Span',df)
text = f"$r={round(r5, 4)}$\n$p={round(p5, 4)}$"
plt.text(6, 40, text,fontsize=16,color='red')
plt.xlabel('Learning Speed')
plt.tight_layout()

ax = plt.subplot(246)
sns.regplot('Learning Speed LB','RAN Letters',df)
text = f"$r={round(r6, 4)}$\n$p={round(p6, 4)}$"
plt.text(7.5, 26, text,fontsize=16,color='red')
plt.xlabel('Learning Speed')
plt.tight_layout()

ax = plt.subplot(247)
sns.regplot('Learning Speed LB','RAN Objects',df)
text = f"$r={round(r7, 4)}$\n$p={round(p7, 4)}$"
plt.text(8.7, 37.5, text,fontsize=16,color='red')
plt.xlabel('Learning Speed')
plt.tight_layout()

ax = plt.subplot(248)
sns.regplot('Learning Speed LB','Phonological Processing',df)
text = f"$r={round(r8, 4)}$\n$p={round(p8, 4)}$"
plt.text(7, 50, text,fontsize=16)
plt.xlabel('Learning Speed')
plt.tight_layout()