import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from nidaqmx import Task
from nidaqmx.constants import TerminalConfiguration
from nidaqmx.system import System
from tkinter import *
from tkinter import messagebox, filedialog, ttk, simpledialog
from PIL import ImageTk, Image
from scipy.optimize import curve_fit
import pandas as pd
import csv
import threading

Documentation_window = Tk()
Documentation_window.title("Documentation")
Documentation_window.iconbitmap('US-marca-principal.ico')
Documentation_window.geometry("790x650")  # Establecer tamaño de la nueva ventana

# Agregar contenido a la nueva ventana (puedes agregar lo que quieras)
label = Label(Documentation_window, text="DOCUMENTATION FOR THE DIODE CHARACTERISTICS APPLICATION")
label.pack(padx=10, pady=10)

# We add a scrollbar
scrollbar = Scrollbar(Documentation_window)
scrollbar.pack(side=RIGHT, fill=Y)

# Create the text in functionalities
doctext = Text(Documentation_window, height=50, yscrollcommand=scrollbar.set)
doctext.pack(padx=10, pady=10, fill='both', expand=True)
scrollbar.config(command=doctext.yview)

# Open and read file
docopen = open('Diodedocumentation.txt', 'r')
docread = docopen.read()
doctext.insert(END, docread)
docopen.close()

Documentation_window.mainloop()