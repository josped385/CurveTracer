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

Help_window = Tk()
Help_window.title("Help")
Help_window.iconbitmap('US-marca-principal.ico')
Help_window.geometry("790x650")  # Establecer tamaño de la nueva ventana
Help_window.resizable(False, False)

helpnotebook = ttk.Notebook(Help_window)
helpnotebook.pack(padx=0, pady=0, fill='both', expand=True)
schematicnotebook = Frame(helpnotebook)   # first page, which would get widgets gridded into it
pinoutnotebook = Frame(helpnotebook)   # second page
pinoutmcnotebook = Frame(helpnotebook)
strinvnotebook = Frame(helpnotebook)
functionalitiesnotebook = Frame(helpnotebook)
helpnotebook.add(schematicnotebook, text='Schematic')
helpnotebook.add(pinoutnotebook, text='741 OpAmp Pinout')
helpnotebook.add(pinoutmcnotebook, text='PinOut of MC4007UBE')
helpnotebook.add(strinvnotebook, text='Extra Parameters')
helpnotebook.add(functionalitiesnotebook, text='Functionalities of the app')


# Agregar contenido a la nueva ventana (puedes agregar lo que quieras)
label = Label(schematicnotebook, text="SCHEMATIC OF THE MOSFET CIRCUIT IMPLEMENTED IN NI MYDAQ")
label.pack(padx=10, pady=10)

# Cargar y redimensionar la imagen del esquemático del diodo
MOSFETSchematicHelp = Image.open("MOSFETCharacteristicsSchematic.png")  # Cambia por la ruta de tu imagen
MOSFETSchematicHelp_resized = MOSFETSchematicHelp.resize((417, 276))  # Cambia el tamaño (ancho, alto)
MOSFETSchematicHelp_photo = ImageTk.PhotoImage(MOSFETSchematicHelp_resized)
MOSFETSchematicHelp_label = Label(schematicnotebook, image=MOSFETSchematicHelp_photo)
MOSFETSchematicHelp_label.pack(padx=10, pady=95)  # Cambia la posición (x, y)

# Cargar y redimensionar la imagen de US
USHelp = Image.open("US-marca-principal.png")  # Cambia por la ruta de tu imagen
USHelp_resized = USHelp.resize((100, 90))  # Cambia el tamaño (ancho, alto)
USHelp_photo = ImageTk.PhotoImage(USHelp_resized)
USHelp_label = Label(schematicnotebook, image=USHelp_photo)
USHelp_label.pack(padx=10, pady=10)  # Cambia la posición (x, y)

# Agregar contenido a la nueva ventana (puedes agregar lo que quieras)
label2 = Label(pinoutnotebook, text="PINOUT OF THE 741 OPAMP IMPLEMENTED IN THE CIRCUIT")
label2.pack(padx=10, pady=10)

# Cargar y redimensionar la imagen del pinout del 741
PinoutHelp = Image.open("MOSFETCharacteristics741PinOut.png")  # Cambia por la ruta de tu imagen
PinoutHelp_resized = PinoutHelp.resize((484, 205))  # Cambia el tamaño (ancho, alto)
PinoutHelp_photo = ImageTk.PhotoImage(PinoutHelp_resized)
PinoutHelp_label = Label(pinoutnotebook, image=PinoutHelp_photo)
PinoutHelp_label.pack(padx=10, pady=130)  # Cambia la posición (x, y)

# Cargar y redimensionar la imagen de US
USHelp_label2 = Label(pinoutnotebook, image=USHelp_photo)
USHelp_label2.pack(padx=10, pady=10)  # Cambia la posición (x, y)

# Agregar contenido a la nueva ventana (puedes agregar lo que quieras)
label3 = Label(pinoutmcnotebook, text="PINOUT OF THE MC4007UBE CHIP IMPLEMENTED IN THE CIRCUIT")
label3.pack(padx=10, pady=10)

# Cargar y redimensionar la imagen del pinout del 741
PinoutmcHelp = Image.open("MOSFETCharacteristicsChip4007.png")  # Cambia por la ruta de tu imagen
PinoutmcHelp_resized = PinoutmcHelp.resize((347, 247))  # Cambia el tamaño (ancho, alto)
PinoutmcHelp_photo = ImageTk.PhotoImage(PinoutmcHelp_resized)
PinoutmcHelp_label = Label(pinoutmcnotebook, image=PinoutmcHelp_photo)
PinoutmcHelp_label.pack(padx=10, pady=110)  # Cambia la posición (x, y)

# Cargar y redimensionar la imagen de US
USHelp_label3 = Label(pinoutmcnotebook, image=USHelp_photo)
USHelp_label3.pack(padx=10, pady=10)  # Cambia la posición (x, y)

# Ahora ponemos los Extra parameters
label5 = Label(strinvnotebook, text="OHMIC AND SATURATION PARAMETERS")
label5.pack()

# Cargar y redimensionar la imagen del pinout del 741
EQ1Help = Image.open("MOSFETCharacteristicsUOTHETAVTO.png")  # Cambia por la ruta de tu imagen
EQ1Help_resized = EQ1Help.resize((383, 35))  # Cambia el tamaño (ancho, alto)
EQ1Help_photo = ImageTk.PhotoImage(EQ1Help_resized)
EQ1Help_label = Label(strinvnotebook, image=EQ1Help_photo)
EQ1Help_label.pack(padx=10, pady=10)  # Cambia la posición (x, y)

# Label para Str Inv parameters
label6 = Label(strinvnotebook, text="STRONG INVERSION PARAMETERS")
label6.pack()

# Cargar y redimensionar la imagen del pinout del 741
EQ2Help = Image.open("MOSFETCharacteristicsStrInv.png")  # Cambia por la ruta de tu imagen
EQ2Help_resized = EQ2Help.resize((360, 406))  # Cambia el tamaño (ancho, alto)
EQ2Help_photo = ImageTk.PhotoImage(EQ2Help_resized)
EQ2Help_label = Label(strinvnotebook, image=EQ2Help_photo)
EQ2Help_label.pack(padx=10, pady=5)  # Cambia la posición (x, y)

# Cargar y redimensionar la imagen de US
USHelp_label3 = Label(strinvnotebook, image=USHelp_photo)
USHelp_label3.pack(padx=10, pady=7)  # Cambia la posición (x, y)

# Agregar contenido a la nueva ventana (puedes agregar lo que quieras)
label4 = Label(functionalitiesnotebook, text="HELP FOR THE FUNCTIONALITIES OF THE MOSFET CHARACTERISTICS APPLICATION")
label4.pack(padx=10, pady=10)

# Create a scrollbar for fucntionalities
scrollbar = Scrollbar(functionalitiesnotebook)
scrollbar.pack(side=RIGHT, fill=Y)

# Create the text in functionalities
functext = Text(functionalitiesnotebook, height=50, yscrollcommand=scrollbar.set)
functext.pack(padx=10, pady=10, fill='both', expand=True)

# Open and read file
funcopen = open('MOSFETfunctionalities.txt', 'r')
funcread = funcopen.read()
functext.insert(END, funcread)
funcopen.close()
scrollbar.config(command=functext.yview)

Help_window.mainloop()
