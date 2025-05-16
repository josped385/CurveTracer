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
from subprocess import call

def Open_DiodeCharacteristics():
    call(['python', 'DiodeCharacteristics.py'])

def Open_DiodeHelp():
    call(['python', 'DiodeHelp.py'])

def Open_DiodeDocumentation():
    call(['python', 'DiodeDocumentation.py'])

def Open_MOSFETCharacteristics():
    call(['python', 'MOSFETCharacteristics.py'])

def Open_MOSFETHelp():
    call(['python', 'MOSFETHelp.py'])

def Open_MOSFETDocumentation():
    call(['python', 'MOSFETDocumentation.py'])

root = Tk()
root.title("Curve Tracer Application")
root.iconbitmap('US-marca-principal.ico')
root.geometry("500x400")
root.resizable(False, False)

# Cargar y redimensionar la imagen de US
US = Image.open("US-marca-principal.png")  # Cambia por la ruta de tu imagen
US_resized = US.resize((100, 90))  # Cambia el tamaño (ancho, alto)
US_photo = ImageTk.PhotoImage(US_resized)
US_label = Label(root, image=US_photo)
US_label.place(x=200, y=50)  # Cambia la posición (x, y)

# Crear botón para Open Diode Characteristics
OpenDiodeCharacteristics = Button(root, text="Open Diode Characteristics", command=Open_DiodeCharacteristics)
OpenDiodeCharacteristics.place(x=50 , y=220)

# Crear botón para Open Diode Help
OpenDiodeHelp = Button(root, text="Open Diode Help", command=Open_DiodeHelp)
OpenDiodeHelp.place(x=70, y=270)

# Crear botón para Open Diode Documentation
OpenDiodeDocumentation = Button(root, text="Open Diode Documentation", command=Open_DiodeDocumentation)
OpenDiodeDocumentation.place(x=45 , y=320)

# Crear botón para Open MOSFET Characteristics
OpenMOSFETCharacteristics = Button(root, text="Open MOSFET Characteristics", command=Open_MOSFETCharacteristics)
OpenMOSFETCharacteristics.place(x=300 , y=220)

# Crear botón para Open MOSFET Help
OpenMOSFETHelp = Button(root, text="Open MOSFET Help", command=Open_MOSFETHelp)
OpenMOSFETHelp.place(x=320, y=270)

# Crear botón para Open MOSFET Documentation
OpenMOSFETDocumentation = Button(root, text="Open MOSFET Documentation", command=Open_MOSFETDocumentation)
OpenMOSFETDocumentation.place(x=295 , y=320)

root.mainloop()