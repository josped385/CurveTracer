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
functionalitiesnotebook = Frame(helpnotebook)
helpnotebook.add(schematicnotebook, text='Schematic')
helpnotebook.add(pinoutnotebook, text='741 OpAmp Pinout')
helpnotebook.add(functionalitiesnotebook, text='Functionalities of the app')


# Agregar contenido a la nueva ventana (puedes agregar lo que quieras)
label = Label(schematicnotebook, text="SCHEMATIC OF THE DIODE CIRCUIT IMPLEMENTED IN NI MYDAQ")
label.pack(padx=10, pady=10)

# Cargar y redimensionar la imagen del esquemático del diodo
DiodeSchematicHelp = Image.open("DiodeCharacteristicsSchematic.png")  # Cambia por la ruta de tu imagen
DiodeSchematicHelp_resized = DiodeSchematicHelp.resize((373, 214))  # Cambia el tamaño (ancho, alto)
DiodeSchematicHelp_photo = ImageTk.PhotoImage(DiodeSchematicHelp_resized)
DiodeSchematicHelp_label = Label(schematicnotebook, image=DiodeSchematicHelp_photo)
DiodeSchematicHelp_label.pack(padx=10, pady=125)  # Cambia la posición (x, y)

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
PinoutHelp = Image.open("DiodeCharacteristics741PinOut.png")  # Cambia por la ruta de tu imagen
PinoutHelp_resized = PinoutHelp.resize((484, 205))  # Cambia el tamaño (ancho, alto)
PinoutHelp_photo = ImageTk.PhotoImage(PinoutHelp_resized)
PinoutHelp_label = Label(pinoutnotebook, image=PinoutHelp_photo)
PinoutHelp_label.pack(padx=10, pady=130)  # Cambia la posición (x, y)

# Cargar y redimensionar la imagen de US
USHelp_label2 = Label(pinoutnotebook, image=USHelp_photo)
USHelp_label2.pack(padx=10, pady=10)  # Cambia la posición (x, y)

# Agregar contenido a la nueva ventana (puedes agregar lo que quieras)
label3 = Label(functionalitiesnotebook, text="HELP FOR THE FUNCTIONALITIES OF THE DIODE CHARACTERISTICS APPLICATION")
label3.pack(padx=10, pady=10)

# Create a scrollbar for fucntionalities
scrollbar = Scrollbar(functionalitiesnotebook)
scrollbar.pack(side=RIGHT, fill=Y)

# Create the text in functionalities
functext = Text(functionalitiesnotebook, height=50, yscrollcommand=scrollbar.set)
functext.pack(padx=10, pady=10, fill='both', expand=True)

# Open and read file
funcopen = open('Diodefunctionalities.txt', 'r')
funcread = funcopen.read()
functext.insert(END, funcread)
funcopen.close()
scrollbar.config(command=functext.yview)

Help_window.mainloop()