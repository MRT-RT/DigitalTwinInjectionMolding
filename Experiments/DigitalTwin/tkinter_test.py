# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:42:32 2022

@author: LocalAdmin
"""

import tkinter as tk
import time

master = tk.Tk()

Q_target = 0.0

w = tk.Scale(master, from_=26, to=28,length=1000,width=100,
             orient='vertical',digits=3,label='Durchmesser_innen',
             resolution=0.1, tickinterval=0.5)

w.pack()
master.mainloop()

# while True:
#     master.update_idletasks()
#     master.update()
#     time.sleep(2)
#     print(w.get())
# mainloop()