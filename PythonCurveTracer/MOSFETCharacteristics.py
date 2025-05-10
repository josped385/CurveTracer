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
from scipy.optimize import curve_fit, least_squares
import pandas as pd
import csv
import threading

Documentation_window = Help_window = Curvefitting_window = StrInv_window = StrInv2_window = None
Vsteps_val = V_O = I_D = IOff_val = R_val = V_G_mean = V_D = V_G = None
VDmin_val = VDmax_val = VGmin_val = VGmax_val = None
IVfig = VOVDfig = IVax = IVcanvas = VOVDax = VOVDcanvas = xaxis = None
sat_line_IVp = sat_line_IVn = sat_line_VOVDp = sat_line_VOVDn = None
cad_line_IV = cad_line_VOVD = None
V_fit = Id_fit = V_D_filtered = V_O_filtered = V_G_filtered = I_D_filtered = None
IVresiduesfig = VOVDresiduesfig = IVresiduesax = VOVDresiduesax = None
Vd = sweep_val = VSB_val = UO_val = UO2_val = THETA_val = THETA2_val = VTO_val = VTO2_val = None
stop_event = threading.Event()

def ChechNImyDAQ():
    # Crear un objeto System
    system = System.local()
    # Listar y verificar los dispositivos conectados
    devices = system.devices
    if len(devices) == 0:
        CheckNImyDAQ_label.config(bg="red")
        messagebox.showerror('Error', 'The NI device has not been detected. Check the connectivity, and if the error persists, consider restarting Python.')
    else:
        CheckNImyDAQ_label.config(bg="green")

def Open_help():
    global Help_window

    if Help_window and Help_window.winfo_exists():
        Help_window.lift()
        Help_window.focus_force()  # Trae la ventana al frente y la pone en foco
    else:
        # Crear una nueva ventana
        Help_window = Toplevel(root)
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

def Open_documentation():
    global Documentation_window

    if Documentation_window and Documentation_window.winfo_exists():
        Documentation_window.lift()
        Documentation_window.focus_force()
    else:
        # Crear una nueva ventana
        Documentation_window = Toplevel(root)
        Documentation_window.title("Documentation")
        Documentation_window.iconbitmap('US-marca-principal.ico')
        Documentation_window.geometry("790x650")  # Establecer tamaño de la nueva ventana

        # Agregar contenido a la nueva ventana (puedes agregar lo que quieras)
        label = Label(Documentation_window, text="DOCUMENTATION FOR THE MOSFET CHARACTERISTICS APPLICATION")
        label.pack(padx=10, pady=10)

        # We add a scrollbar
        scrollbar = Scrollbar(Documentation_window)
        scrollbar.pack(side=RIGHT, fill=Y)

        # Create the text in functionalities
        doctext = Text(Documentation_window, height=50, yscrollcommand=scrollbar.set)
        doctext.pack(padx=10, pady=10, fill='both', expand=True)
        scrollbar.config(command=doctext.yview)

        # Open and read file
        docopen = open('MOSFETdocumentation.txt', 'r')
        docread = docopen.read()
        doctext.insert(END, docread)
        docopen.close()

def Runmeasurement():

    global Vsteps_val, V_O, I_D, IVfig, IVax, IVcanvas, VOVDax, VOVDcanvas, VOVDfig, IOff_val, R_val
    global V_G_mean, V_D, V_G, VDmin_val, VDmax_val, VGmin_val, VGmax_val, xaxis
    # --- Obtener parámetros de los Entry ---
    try:
        VDmin_val = float(VDmin.get())        # Voltaje mínimo para VD
        VDmax_val = float(VDmax.get())        # Voltaje máximo para VD
        VDsteps_val = int(VDsteps.get())      # Número de pasos para VD
        VGmin_val = float(VGmin.get())        # Voltaje fijo para VG
        VGmax_val = float(VGmax.get())        # Voltaje fijo para VG
        VGsteps_val = int(VGsteps.get())      # Pasos (fijo en este caso)
        navg = int(NAveraging.get())          # Número de mediciones para promediar
        IOff_val = float(IOff.get())              # Corriente de offset
        R_val = float(R.get())
    except Exception as e:
        messagebox.showerror("Error", f"Revise the input parameters:\n{e}")
        return
    
    # Parámetros del dispositivo
    device_id = DeviceID.get()
    vd_channel = VD_value.get()
    vg_channel = VG_value.get()
    vo_channel = VO_value.get()

    # Ejes de los gráficos
    xaxis = XAxis_value.get()

    if vd_channel == vg_channel:
        messagebox.showerror('Error', 'The VD and VG channels must be different.')
        return
    elif CheckNImyDAQ_label.cget("bg") == "red":
        messagebox.showerror('Error', 'Turn on the NI device before simulating.')
        return
    elif VDmin_val < -10:
        messagebox.showerror('Error', 'Channels configured for -10 to +10 Volts.')
        return
    elif VDmax_val > 10:
        messagebox.showerror('Error', 'Channels configured for -10 to +10 Volts.')
        return
    elif VGmin_val < -10:
        messagebox.showerror('Error', 'Channels configured for -10 to +10 Volts.')
        return
    elif VGmax_val > 10:
        messagebox.showerror('Error', 'Channels configured for -10 to +10 Volts.')
        return
    elif VDsteps_val < 0:
        messagebox.showerror('Error', 'VDsteps must be a positive value and an integer.')
        return
    elif VGsteps_val < 0:
        messagebox.showerror('Error', 'VGsteps must be a positive value and an integer.')
        return
    
    if sweep_val.get():
        # Reiniciamos el evento (aseguramos que no esté marcado)
        stop_event.clear()

        # Crear vector de barrido para VD y VG
        V_D = np.linspace(VDmin_val, VDmax_val, VDsteps_val)

        V_G = np.linspace(VGmin_val, VGmax_val, VGsteps_val)

        # Reservar arrays para resultados
        VO = np.zeros((len(V_G), len(V_D)))
        ID = np.zeros_like(VO)
        step = 0
        if xaxis == 'V_GS':
            Ioff = np.linspace(IOff_val, IOff_val, VGsteps_val)
        else:
            Ioff = np.linspace(IOff_val, IOff_val, VDsteps_val)
        total_steps = navg * VDsteps_val * VGsteps_val
        # --- Realizar mediciones y promediar ---
        with Task() as output_task, Task() as input_task:
            # Configurar canales de salida (VD y VG)
            output_task.ao_channels.add_ao_voltage_chan(f"{device_id}/{vd_channel}")
            output_task.ao_channels.add_ao_voltage_chan(f"{device_id}/{vg_channel}")
            # Configurar canal de entrada (V0) en modo diferencial
            input_task.ai_channels.add_ai_voltage_chan(f"{device_id}/{vo_channel}",
                                                    terminal_config=TerminalConfiguration.DIFF)
            
            progressbar.config(maximum=total_steps)  # O progressbar["maximum"] = total_steps
            # Barrido anidado
            for i, VG in enumerate(V_G):
                for j, VD in enumerate(V_D):
                    if stop_event.is_set():
                        messagebox.showerror('Simulation stopped', 'The simulation has been stopped')
                        return
                    output_task.write([VD, VG])
                    VO[i, j] = input_task.read()
                    ID[i, j] = (VO[i, j] - VD) / R_val
                    # -- Actualizar progreso aquí dentro del navg --
                    step += 1
                    progressbar_var.set(step)
                    root.update_idletasks()
                    root.after(1)

        # --- Graficar los resultados ---
        # Para la gráfica IV (corriente vs voltaje)
        # Crear Frame para la gráfica IV
        IV = Frame(root, width=463, height=333, bg="lightgray")
        IV.place(x=29, y=700-26-333)
        IV.grid_propagate(False)
        # Crear nueva figura
        IVfig = Figure(figsize=(4.5, 3.5), dpi=100)
        IVax = IVfig.add_subplot(111)
        # Añadir la curva actual
        if xaxis == 'V_DS':
            for i, VG in enumerate(V_G):
                IVax.plot(V_D, ID[i], label=f"VG = {VG:.2f} V", linewidth=1)
            IVax.set_xlabel("V_DS [V]", fontsize=10)
        elif xaxis == 'V_GS':
            for j, VD in enumerate(V_D):
                IVax.plot(V_G, ID[:, j], label=f"VD = {VD:.2f} V", linewidth=1)
            IVax.set_xlabel("V_GS [V]", fontsize=10)
        IVax.set_title("MOSFET I-V Characteristic", fontsize=10)
        IVax.set_ylabel("I_D [A]", fontsize=10)
        IVax.grid(True, linestyle='--', linewidth=0.5, color="gray")
        IVfig.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)
        IVcanvas = FigureCanvasTkAgg(IVfig, master=IV)
        IVcanvas.draw()
        IVcanvas.get_tk_widget().pack(fill="both", expand=True)
        IVax.legend()

        VOVD = Frame(root, width=463, height=333, bg="lightgray")
        VOVD.place(x=604, y=700-26-333)
        VOVD.grid_propagate(False)
        VOVDfig = Figure(figsize=(4.5, 3.5), dpi=100)
        VOVDax = VOVDfig.add_subplot(111)
        if xaxis == 'V_DS':
            for i, VG in enumerate(V_G):
                VOVDax.plot(V_D, VO[i], label=f"VG = {VG:.2f} V", linewidth=1)
            VOVDax.set_xlabel("V_DS [V]", fontsize=10)
        elif xaxis == 'V_GS':
            for j, VD in enumerate(V_D):
                VOVDax.plot(V_G, VO[:, j], label=f"VD = {VD:.2f} V", linewidth=1)
            VOVDax.set_xlabel("V_GS [V]", fontsize=10)
        VOVDax.set_title("Output Voltage", fontsize=10)
        VOVDax.set_ylabel("V_O [V]", fontsize=10)
        VOVDax.grid(True, linestyle='--', linewidth=0.5, color="gray")
        VOVDfig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.2)
        VOVDcanvas = FigureCanvasTkAgg(VOVDfig, master=VOVD)
        VOVDcanvas.draw()
        VOVDcanvas.get_tk_widget().pack(fill="both", expand=True)
        VOVDax.legend()
    else:
        # Reiniciamos el evento (aseguramos que no esté marcado)
        stop_event.clear()

        # Definimos Vsteps
        Vsteps_val = np.maximum(VDsteps_val, VGsteps_val)

        # Crear vector de barrido para VD y VG
        V_D = np.linspace(VDmin_val, VDmax_val, Vsteps_val)

        V_G = np.linspace(VGmin_val, VGmax_val, Vsteps_val)

        V_G_mean = np.mean(V_G)

        # --- Realizar mediciones y promediar ---
        # Creamos un arreglo para almacenar cada medición
        all_V_O = np.zeros((navg, Vsteps_val))
        with Task() as output_task, Task() as input_task:
            # Configurar canales de salida (VD y VG)
            output_task.ao_channels.add_ao_voltage_chan(f"{device_id}/{vd_channel}")
            output_task.ao_channels.add_ao_voltage_chan(f"{device_id}/{vg_channel}")
            # Configurar canal de entrada (V0) en modo diferencial
            input_task.ai_channels.add_ai_voltage_chan(f"{device_id}/{vo_channel}",
                                                    terminal_config=TerminalConfiguration.DIFF)
            
            total_steps = navg * Vsteps_val
            progressbar.config(maximum=total_steps)  # O progressbar["maximum"] = total_steps

            # Realizar 'navg' mediciones
            for avg in range(navg):
                for i in range(Vsteps_val):
                    if stop_event.is_set():
                        messagebox.showerror('Simulation stopped', 'The simulation has been stopped')
                        return
                    output_task.write([V_D[i], V_G[i]])
                    data = input_task.read()
                    all_V_O[avg, i] = data
                    # Calcular el progreso global:
                    current_step = avg * Vsteps_val + i + 1
                    progressbar_var.set(current_step)
                    root.update_idletasks()
                    root.after(1)

        # Promediar las mediciones
        Ioff = np.linspace(IOff_val, IOff_val, Vsteps_val)
        V_O = np.mean(all_V_O, axis=0)
        #R_val = float(R.get())
        I_D = (V_O - V_D)/R_val - Ioff

        # --- Graficar los resultados ---
        # Para la gráfica IV (corriente vs voltaje)
        # Crear Frame para la gráfica IV
        IV = Frame(root, width=463, height=333, bg="lightgray")
        IV.place(x=29, y=700-26-333)
        IV.grid_propagate(False)
        # Crear nueva figura
        IVfig = Figure(figsize=(4.5, 3.5), dpi=100)
        IVax = IVfig.add_subplot(111)
        # Añadir la curva actual
        if xaxis == 'V_DS':
            IVax.plot(V_D, I_D, label="MOSFET Current", linewidth=1)
            IVax.set_xlabel("V_DS [V]", fontsize=10)
        elif xaxis == 'V_GS':
            IVax.plot(V_G, I_D, label="MOSFET Current", linewidth=1)
            IVax.set_xlabel("V_GS [V]", fontsize=10)
        IVax.set_title("MOSFET I-V Characteristic", fontsize=10)
        IVax.set_ylabel("I_D [A]", fontsize=10)
        IVax.grid(True, linestyle='--', linewidth=0.5, color="gray")
        IVfig.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)
        IVcanvas = FigureCanvasTkAgg(IVfig, master=IV)
        IVcanvas.draw()
        IVcanvas.get_tk_widget().pack(fill="both", expand=True)
        IVax.legend()

        VOVD = Frame(root, width=463, height=333, bg="lightgray")
        VOVD.place(x=604, y=700-26-333)
        VOVD.grid_propagate(False)
        VOVDfig = Figure(figsize=(4.5, 3.5), dpi=100)
        VOVDax = VOVDfig.add_subplot(111)
        if xaxis == 'V_DS':
            VOVDax.plot(V_D, V_O, label="Output Voltage", linewidth=1)
            VOVDax.set_xlabel("V_DS [V]", fontsize=10)
        elif xaxis == 'V_GS':
            VOVDax.plot(V_G, V_O, label="Output Voltage", linewidth=1)
            VOVDax.set_xlabel("V_GS [V]", fontsize=10)
        VOVDax.set_title("Output Voltage", fontsize=10)
        VOVDax.set_ylabel("V_O [V]", fontsize=10)
        VOVDax.grid(True, linestyle='--', linewidth=0.5, color="gray")
        VOVDfig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.2)
        VOVDcanvas = FigureCanvasTkAgg(VOVDfig, master=VOVD)
        VOVDcanvas.draw()
        VOVDcanvas.get_tk_widget().pack(fill="both", expand=True)
        VOVDax.legend()

def Stop():
    stop_event.set()

def Curvefitting():

    if sweep_val.get():
        messagebox.showerror('Error', 'Curve fitting is not available for sweep data.')
        return
    else:
        def Calculate():
            global V_fit, Id_fit, V_D_filtered, V_O_filtered, V_G_filtered, I_D_filtered

            if V_D is None or I_D is None or V_O is None:  # Si está definida pero vacía
                messagebox.showerror('Error', 'Run a measurement in order to fit the curve.')
                return
            
            # Definir la función modelo del MOSFET (piecewise)
            def mosfet_model(vars, Vth, K, lam):
                # vars es una tupla (Vg, Vd)
                Vg, Vd = vars
                Vg = np.array(Vg)
                Vd = np.array(Vd)
                Id_est = np.zeros_like(Vg)
                
                # Definir máscaras para cada región de operación
                mask_cutoff = (Vg < Vth)
                mask_triode = (Vg >= Vth) & (Vd < (Vg - Vth))
                mask_sat    = (Vg >= Vth) & (Vd >= (Vg - Vth))
                
                # Región de corte
                Id_est[mask_cutoff] = 0.0
                # Región triodo
                Id_est[mask_triode] = K * ((Vg[mask_triode] - Vth) * Vd[mask_triode] 
                                        - 0.5 * (Vd[mask_triode] ** 2))
                # Región de saturación (con channel-length modulation)
                Id_est[mask_sat] = 0.5 * K * (Vg[mask_sat] - Vth) ** 2 * (1 + lam * Vd[mask_sat])
                return Id_est
            
            # Seleccionar los datos que cumplen la condición
            if xaxis == 'V_DS':
                valid_idx = V_O < 10
                V_G_filtered = V_G[valid_idx]
                V_D_filtered = V_D[valid_idx]
                I_D_filtered = I_D[valid_idx]
                V_O_filtered = V_O[valid_idx]
            elif xaxis == 'V_GS':
                valid_idx = V_O < 10
                V_G_filtered = V_G[valid_idx]
                V_D_filtered = V_D[valid_idx]
                I_D_filtered = I_D[valid_idx]
                V_O_filtered = V_O[valid_idx]
            
            # Establecer parámetros iniciales para el ajuste
            p0 = [1.0, 1e-4, 0.01]  # [Vth_inicial, K_inicial, lambda_inicial]
            
            # Ajuste con scipy.optimize.curve_fit
            popt, pcov = curve_fit(mosfet_model, (V_G_filtered, V_D_filtered), I_D_filtered, p0=p0)
            Vth_fit, K_fit, lam_fit = popt
            
            # Calcular la respuesta ajustada
            Id_fit = mosfet_model((V_G_filtered, V_D_filtered), *popt)
            
            # Calcular el coeficiente de determinación R²
            residuals = I_D_filtered - Id_fit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((I_D_filtered - np.mean(I_D_filtered))**2)
            r2 = 1 - (ss_res / ss_tot)

            # Calcular errores (desviación estándar de cada parámetro)
            perr = np.sqrt(np.diag(pcov))
            err_Vth, err_K, err_lam = perr

            # Asignar resultados a los campos de la interfaz
            Vth_val.insert(0, f"{Vth_fit: .7g}")
            K_val.insert(0, f"{K_fit: .7g}")
            lambda_val.insert(0, f"{lam_fit: .7g}")
            errorVth.insert(0, f"{err_Vth: .7g}")
            errorK.insert(0, f"{err_K: .7g}")
            errorlambda.insert(0, f"{err_lam: .7g}")
            R2.insert(0, f"{r2: .5g}")

            V_fit = R_val*Id_fit + V_D_filtered

            # Añadir la curva actual
            if XAxis_value.get() == 'V_DS':
                IVax.plot(V_D_filtered, Id_fit, label="Ifit", linewidth=1)
                IVax.set_xlabel("V_DS [V]", fontsize=10)
            elif XAxis_value.get() == 'V_GS':
                IVax.plot(V_G_filtered, Id_fit, label="Ifit", linewidth=1)
                IVax.set_xlabel("V_GS [V]", fontsize=10)
            IVax.legend()
            IVcanvas.draw()

            # Añadir la curva actual
            if XAxis_value.get() == 'V_DS':
                VOVDax.plot(V_D_filtered, V_fit, label="Vfit", linewidth=1)
                VOVDax.set_xlabel("V_DS [V]", fontsize=10)
            elif XAxis_value.get() == 'V_GS':
                VOVDax.plot(V_G_filtered, V_fit, label="Vfit", linewidth=1)
                VOVDax.set_xlabel("V_GS [V]", fontsize=10)
            VOVDax.legend()
            VOVDcanvas.draw()

        def Export_fit_data():
            # Crear un diccionario con los nombres y los valores actuales de cada Entry.
            data = {
                "Vth": Vth_val.get(),
                "K": K_val.get(),
                "lambda": lambda_val.get(),
                "ErrorVth": errorVth.get(),
                "ErrorK": errorK.get(),
                "Errorlambda": errorlambda.get(),
                "R2": R2.get(),
            }
            
            # Abrir un cuadro de diálogo "Guardar como..."
            filename = filedialog.asksaveasfilename(
                title="Save data in CSV",
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
            )
            if not filename:
                return  # Se canceló la selección

            try:
                with open(filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    # Escribir la primera fila con los nombres (cabeceras)
                    headers = list(data.keys())
                    writer.writerow(headers)
                    # Escribir la segunda fila con los datos
                    values = [data[key] for key in headers]
                    writer.writerow(values)
                messagebox.showinfo("Saved", f"Data saved in:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Error ocurred when saving file:\n{e}")

        def Get_residues():
            global IVresiduesfig, VOVDresiduesfig, IVresiduesax, VOVDresiduesax
            # Comprobar que se han definido los datos necesarios
            if (V_D is None or V_G is None or V_fit is None or Id_fit is None):
                messagebox.showerror("Error", "We can't calculate the residues because the necessary parameters are not defined.")
                return

            # Calcular residuos para V_O vs V
            residues_V = V_O_filtered - V_fit

            # Calcular residuos para I vs V
            residues_I = I_D_filtered - Id_fit

            VOVDresidues = Frame(Curvefitting_window, width=400, height=100, bg="lightgray")
            VOVDresidues.place(x=550, y=225)
            VOVDresidues.grid_propagate(False)
            VOVDresiduesfig = Figure(figsize=(4.5, 2.2), dpi=100)
            VOVDresiduesax = VOVDresiduesfig.add_subplot(111)
            if XAxis_value.get() == 'V_DS':
                VOVDresiduesax.plot(V_D_filtered, residues_V, linewidth=1)
                VOVDresiduesax.set_xlabel("V_DS [V]", fontsize=10)
            elif XAxis_value.get() == 'V_GS':
                VOVDresiduesax.plot(V_G_filtered, residues_V, linewidth=1)
                VOVDresiduesax.set_xlabel("V_GS [V]", fontsize=10)
            VOVDresiduesax.set_title("Residues for V_O vs V", fontsize=10)
            VOVDresiduesax.set_ylabel("Residues for V_O [V]", fontsize=10)
            VOVDresiduesax.grid(True, linestyle='--', linewidth=0.5, color="gray")
            VOVDresiduesfig.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)
            VOVDresiduescanvas = FigureCanvasTkAgg(VOVDresiduesfig, master=VOVDresidues)
            VOVDresiduescanvas.draw()
            VOVDresiduescanvas.get_tk_widget().pack(fill="both", expand=True)

            IVresidues = Frame(Curvefitting_window, width=400, height=100, bg="lightgray")
            IVresidues.place(x=550, y=0)
            IVresidues.grid_propagate(False)
            IVresiduesfig = Figure(figsize=(4.5, 2.2), dpi=100)
            IVresiduesax = IVresiduesfig.add_subplot(111)
            if XAxis_value.get() == 'V_DS':
                IVresiduesax.plot(V_D_filtered, residues_I, linewidth=1)
                IVresiduesax.set_xlabel("V_DS [V]", fontsize=10)
            elif XAxis_value.get() == 'V_GS':
                IVresiduesax.plot(V_G_filtered, residues_I, linewidth=1)
                IVresiduesax.set_xlabel("V_GS [V]", fontsize=10)
            IVresiduesax.set_title("Residues for I vs V", fontsize=10)
            IVresiduesax.set_ylabel("Residues for I [A]", fontsize=10)
            IVresiduesax.grid(True, linestyle='--', linewidth=0.5, color="gray")
            IVresiduesfig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.2)
            IVresiduescanvas = FigureCanvasTkAgg(IVresiduesfig, master=IVresidues)
            IVresiduescanvas.draw()
            IVresiduescanvas.get_tk_widget().pack(fill="both", expand=True)
        
        def Save_IVres():
            # Abre un diálogo para guardar el archivo
            filetypes = [("PNG Image", "*.png"),
                        ("JPEG Image", "*.jpg"),
                        ("SVG Image", "*.svg"),
                        ("PDF Document", "*.pdf"),
                        ("All Files", "*.*")]
            filename = filedialog.asksaveasfilename(
                title="Save Figure As",
                defaultextension=".png",
                filetypes=filetypes)
            
            if filename:
                try:
                    IVresiduesfig.savefig(filename)
                    messagebox.showinfo("Save Figure", f"Figure saved as:\n{filename}")
                except Exception as e:
                    messagebox.showerror("Save Figure", f"Error saving figure:\n{e}")
        
        def Save_VOVres():
            # Abre un diálogo para guardar el archivo
            filetypes = [("PNG Image", "*.png"),
                        ("JPEG Image", "*.jpg"),
                        ("SVG Image", "*.svg"),
                        ("PDF Document", "*.pdf"),
                        ("All Files", "*.*")]
            filename = filedialog.asksaveasfilename(
                title="Save Figure As",
                defaultextension=".png",
                filetypes=filetypes)
            
            if filename:
                try:
                    VOVDresiduesfig.savefig(filename)
                    messagebox.showinfo("Save Figure", f"Figure saved as:\n{filename}")
                except Exception as e:
                    messagebox.showerror("Save Figure", f"Error saving figure:\n{e}")

        def Export_IV_res_Data():
            # Se asume que IVax es un objeto Axes de Matplotlib (por ejemplo, global)
            lines = IVresiduesax.get_lines()
            if not lines:
                messagebox.showerror("Error", "Couldn't find any data in Diode I-V Characteristic figure.")
                return

            data_dict = {}
            # Recorrer cada línea (curva)
            for idx, line in enumerate(lines):
                # Obtener la etiqueta de la curva; si no tiene etiqueta útil, se asigna un nombre por defecto.
                label = line.get_label()
                if not label or label.startswith('_'):
                    label = f"Curve_{idx+1}"
                # Extraer los datos de x e y y convertirlos en un arreglo de NumPy
                x_data = np.array(line.get_xdata())
                y_data = np.array(line.get_ydata())
                # Guardar en el diccionario usando como claves el nombre de la curva + "_x" y "_y"
                data_dict[f"{label}_x"] = pd.Series(x_data)
                data_dict[f"{label}_y"] = pd.Series(y_data)

            # Crear un DataFrame con las Series; Pandas rellenará con NaN las filas faltantes
            df = pd.DataFrame(data_dict)

            # Abrir un diálogo para seleccionar dónde guardar el CSV
            filename = filedialog.asksaveasfilename(
                title="Save data as CSV",
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
            )
            if filename:
                try:
                    df.to_csv(filename, index=False)
                    messagebox.showinfo("Saved", f"Data saved in:\n{filename}")
                except Exception as e:
                    messagebox.showerror("Error", f"Error when saving file:\n{e}")
        
        def Export_VV_res_Data():
            # Se asume que IVax es un objeto Axes de Matplotlib (por ejemplo, global)
            lines = VOVDresiduesax.get_lines()
            if not lines:
                messagebox.showerror("Error", "Couldn't find any data in Diode I-V Characteristic figure.")
                return

            data_dict = {}
            # Recorrer cada línea (curva)
            for idx, line in enumerate(lines):
                # Obtener la etiqueta de la curva; si no tiene etiqueta útil, se asigna un nombre por defecto.
                label = line.get_label()
                if not label or label.startswith('_'):
                    label = f"Curve_{idx+1}"
                # Extraer los datos de x e y y convertirlos en un arreglo de NumPy
                x_data = np.array(line.get_xdata())
                y_data = np.array(line.get_ydata())
                # Guardar en el diccionario usando como claves el nombre de la curva + "_x" y "_y"
                data_dict[f"{label}_x"] = pd.Series(x_data)
                data_dict[f"{label}_y"] = pd.Series(y_data)

            # Crear un DataFrame con las Series; Pandas rellenará con NaN las filas faltantes
            df = pd.DataFrame(data_dict)

            # Abrir un diálogo para seleccionar dónde guardar el CSV
            filename = filedialog.asksaveasfilename(
                title="Save data as CSV",
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
            )
            if filename:
                try:
                    df.to_csv(filename, index=False)
                    messagebox.showinfo("Saved", f"Data saved in:\n{filename}")
                except Exception as e:
                    messagebox.showerror("Error", f"Error when saving file:\n{e}")

        global Curvefitting_window

        if Curvefitting_window and Curvefitting_window.winfo_exists():
            Curvefitting_window.lift()
            Curvefitting_window.focus_force()  # Trae la ventana al frente y la pone en foco
        else:
            # Crear una nueva ventana
            Curvefitting_window = Toplevel(root)
            Curvefitting_window.title("Curve fitting")
            Curvefitting_window.iconbitmap('US-marca-principal.ico')
            Curvefitting_window.geometry("1050x480")  # Establecer tamaño de la nueva ventana
            Curvefitting_window.resizable(False, False)

            # Configurar las columnas para que se expandan proporcionalmente
            for col in range(6):  # Total de columnas: 0 a 4
                InputSignalProperties.grid_columnconfigure(col, weight=1)

            # Configurar las filas para que se expandan proporcionalmente
            for row in range(11):  # Total de filas: 0 a 3
                InputSignalProperties.grid_rowconfigure(row, weight=1)

            # Crear un menú bar
            menubar = Menu(Curvefitting_window)

            # Crear el menú "File" y sus opciones
            file_menu = Menu(menubar, tearoff=0)
            # Crear el submenú para 'Save Figure'
            save_menu = Menu(menubar, tearoff=0)
            save_menu.add_command(label="Save I-V residue plot", command=Save_IVres)
            save_menu.add_command(label="Save V-V residue plot", command=Save_VOVres)
            file_menu.add_cascade(label="Save Figure...", menu=save_menu)
            file_menu.add_command(label="Export Fit Parameters", command=Export_fit_data)
            export_res = Menu(menubar, tearoff=0)
            export_res.add_command(label="Export I-V Residues Data", command=Export_IV_res_Data)
            export_res.add_command(label="Export V-V Residues Data", command=Export_VV_res_Data)
            file_menu.add_cascade(label="Export Data...", menu=export_res)

            # Agregar los menús al menú bar
            menubar.add_cascade(label="Options", menu=file_menu)

            # Configurar el menú de la ventana principal
            Curvefitting_window.config(menu=menubar)

            #Create Calculate button
            Calculate = Button(Curvefitting_window, text="Calculate", command=Calculate)
            Calculate.grid(row=0, columnspan=7)

            #Create Export Fit Parameters button
            #Export = Button(Curvefitting_window, text="Export Fit Parameters", command=Export_fit_data)
            #Export.grid(row=12, columnspan=7)

            #Create Get residues button
            Residues = Button(Curvefitting_window, text='Get residues', command=Get_residues)
            Residues.grid(row=12, columnspan=7)

            # Cargar y redimensionar la ecuación de la intensidad que atraviesa el diodo
            IEquation = Image.open("MOSFETCharacteristicsCurrentEquation.png")  # Cambia por la ruta de tu imagen
            IEquation_resized = IEquation.resize((415, 55))  # Cambia el tamaño (ancho, alto)
            IEquation_photo = ImageTk.PhotoImage(IEquation_resized)
            Curvefitting_window.IEquation_photo = IEquation_photo
            IEquation_label = Label(Curvefitting_window, image=IEquation_photo)
            IEquation_label.grid(row=1, columnspan=7)  # Cambia la posición (x, y)

            # Cargar y redimensionar la ecuación de la intensidad que atraviesa el diodo
            KEquation = Image.open("MOSFETCharacteristicsKEquation.png")  # Cambia por la ruta de tu imagen
            KEquation_resized = KEquation.resize((70, 25))  # Cambia el tamaño (ancho, alto)
            KEquation_photo = ImageTk.PhotoImage(KEquation_resized)
            Curvefitting_window.KEquation_photo = KEquation_photo
            KEquation_label = Label(Curvefitting_window, image=KEquation_photo)
            KEquation_label.grid(row=2, columnspan=7)  # Cambia la posición (x, y)

            # Cargar y redimensionar la ecuación de la intensidad que atraviesa el diodo
            VEquation = Image.open("MOSFETCharacteristicsVoltageEquation.png")  # Cambia por la ruta de tu imagen
            VEquation_resized = VEquation.resize((90, 12))  # Cambia el tamaño (ancho, alto)
            VEquation_photo = ImageTk.PhotoImage(VEquation_resized)
            Curvefitting_window.VEquation_photo = VEquation_photo
            VEquation_label = Label(Curvefitting_window, image=VEquation_photo)
            VEquation_label.grid(row=7, columnspan=7)  # Cambia la posición (x, y)

            # Create label for Vth
            Vth_label = Label(Curvefitting_window, text="Vth (V)", padx=10, pady=10)
            Vth_label.grid(row=8, column=0, sticky="e")

            # Create entry for Vth
            Vth_val = Entry(Curvefitting_window, width=15)
            Vth_val.grid(row=8, column=1, sticky="w")

            # Create label for errorVth
            errorVth_label = Label(Curvefitting_window, text="Error Vth (V)", padx=10, pady=10)
            errorVth_label.grid(row=8, column=5, sticky="e")

            # Create entry for errorVth
            errorVth = Entry(Curvefitting_window, width=15)
            errorVth.grid(row=8, column=6, sticky="w")

            # Create label for K
            K_label = Label(Curvefitting_window, text="K (A/V^2)", padx=10, pady=10)
            K_label.grid(row=9, column=0, sticky="e")

            # Create entry for K
            K_val = Entry(Curvefitting_window, width=15)
            K_val.grid(row=9, column=1, sticky="w")

            # Create label for errorK
            errorK_label = Label(Curvefitting_window, text="Error K (A/V^2)", padx=10, pady=10)
            errorK_label.grid(row=9, column=5, sticky="e")

            # Create entry for errorK
            errorK = Entry(Curvefitting_window, width=15)
            errorK.grid(row=9, column=6, sticky="w")

            # Create label for lambda
            lambda_label = Label(Curvefitting_window, text="lambda (1/V)", padx=10, pady=10)
            lambda_label.grid(row=10, column=0, sticky="e")

            # Create entry for lambda
            lambda_val = Entry(Curvefitting_window, width=15)
            lambda_val.grid(row=10, column=1, sticky="w")

            # Create label for errorlambda
            errorlambda_label = Label(Curvefitting_window, text="Error lambda (1/V)", padx=10, pady=10)
            errorlambda_label.grid(row=10, column=5, sticky="e")

            # Create entry for errorlambda
            errorlambda = Entry(Curvefitting_window, width=15)
            errorlambda.grid(row=10, column=6, sticky="w")

            # Create label for R^2
            R2_label = Label(Curvefitting_window, text="R^2", padx=10, pady=10)
            R2_label.grid(row=11, column=3, sticky="e")

            # Create entry for R^2
            R2 = Entry(Curvefitting_window, width=15)
            R2.grid(row=11, column=4, sticky="w")

def Showsatlimits():
    global sat_line_IVp, sat_line_IVn, sat_line_VOVDp, sat_line_VOVDn, V_D, V_G

    xaxis = XAxis_value.get()

    # Verifica que los vectores V_D, I_D y V_O tengan datos
    if V_D is None or V_G is None:
        messagebox.showerror('Error', 'Run a measurement in order to show the saturation lines.')
        return

    # Verificar que los ejes existan
    if IVax is None or VOVDax is None:
        messagebox.showerror("Error", "Figures are not available.")
        return

    # Si las líneas de saturación ya están representadas, eliminarlas
    if sat_line_IVp is not None:
       # Para las líneas en la gráfica IV
        #for line in sat_line_IVp:
        #    if line in IVax.get_lines():
        #        # Vaciar los datos de la línea
        #        line.remove()
        #        line.set_data([], [])
        #for line in sat_line_IVn:
        #    if line in IVax.get_lines():
        #        line.remove()
        #        line.set_data([], [])
        ## Para las líneas en la gráfica VOVD
        #for line in sat_line_VOVDp:
        #    if line in VOVDax.get_lines():
        #        line.remove()
        #        line.set_data([], [])
        #for line in sat_line_VOVDn:
        #    if line in VOVDax.get_lines():
        #        line.remove()
        #        line.set_data([], [])
        ## Limpiar las variables globales
        #sat_line_IVp = None
        #sat_line_IVn = None
        #sat_line_VOVDp = None
        #sat_line_VOVDn = None
        def eliminar_lineas_saturacion(ejes, etiquetas_saturacion):
            """Elimina todas las líneas de saturación de los ejes dados."""
            for ax in ejes:
                for line in ax.get_lines():
                    if any(label in line.get_label() for label in etiquetas_saturacion):
                        line.remove()
                        line.set_data([], [])

        # Definir los ejes de las gráficas
        ejes = [IVax, VOVDax]
        sat_line_IVp = None
        sat_line_IVn = None
        sat_line_VOVDp = None
        sat_line_VOVDn = None

        # Etiquetas utilizadas para identificar líneas de saturación
        etiquetas_saturacion = ["Isatp", "Isatn", "Vsatp", "Vsatn", "Isat+", "Isat-", "Vsat+", "Vsat-"]

        # Llamar a la función para eliminar las líneas
        eliminar_lineas_saturacion(ejes, etiquetas_saturacion)
    else:
        if sweep_val.get():
            sat_line_VOVDp = VOVDax.axhline(10, color='k', linestyle='--', label='Vsat+')
            sat_line_VOVDn = VOVDax.axhline(-10, color='k', linestyle='--', label='Vsat-')
            Ioff = IOff_val
            if xaxis == 'V_DS':
                for i, VG in enumerate(V_G):
                    I_sat = (10 - V_D)/R_val - Ioff
                    sat_line_IVp = IVax.plot(V_D, I_sat, 'k--', label='Isat+')
                    sat_line_IVn = IVax.plot(V_D, -I_sat, 'k--', label='Isat-')
            else:
                for j, VD in enumerate(V_D):
                    #VD_ref = V_D.min()
                    I_sat = (10 - VD)/R_val - Ioff
                    sat_line_IVp = IVax.hlines(I_sat, V_G.min(), V_G.max(), colors='k', linestyles='--', label='Isat+')
                    sat_line_IVn = IVax.hlines(-I_sat, V_G.min(), V_G.max(), colors='k', linestyles='--', label='Isat-')
        else:
            # Crear los vectores para las líneas de saturación:
            N = Vsteps_val  # Solo necesitamos dos puntos para dibujar una línea (inicio y fin)
            Vsatp = np.linspace(10, 10, N)  # Valores x para la línea positiva
            Vsatn = np.linspace(-10, -10, N)  # Valores x para la línea negativa
            if xaxis == 'V_DS':
                Isat = (10 - V_D)/ R_val - IOff_val
            elif xaxis == 'V_GS':
                Isat = (10 - VDmin_val)/R_val - IOff_val
                Isatp = np.linspace(Isat, Isat, N)
                Isatn = np.linspace(-Isat, -Isat, N)

            # Añadir las líneas de saturación a la gráfica

            if xaxis == 'V_DS':
                sat_line_IVp = IVax.plot(V_D, Isat, 'r--', label="Isat+")
                sat_line_IVn = IVax.plot(V_D, -Isat, 'r--', label="Isat-")

                sat_line_VOVDp = VOVDax.plot(V_D, Vsatp, 'r--', label="Vsat+")
                sat_line_VOVDn = VOVDax.plot(V_D, Vsatn, 'r--', label="Vsat-")
            elif xaxis == 'V_GS':
                sat_line_IVp = IVax.plot(V_G, Isatp, 'r--', label="Isat+")
                sat_line_IVn = IVax.plot(V_G, Isatn, 'r--', label="Isat-")

                sat_line_VOVDp = VOVDax.plot(V_G, Vsatp, 'r--', label="Vsat+")
                sat_line_VOVDn = VOVDax.plot(V_G, Vsatn, 'r--', label="Vsat-")
    
    # (Opcional) Actualizar la leyenda para incluir las nuevas líneas
    IVax.legend()
    VOVDax.legend()

    # Redibujar el canvas para que aparezcan los cambios
    IVcanvas.draw()
    VOVDcanvas.draw()

def Linlog():
    # Configurar escala: si el radiobutton está en "log", se usa escala logarítmica en el eje Y
    if scale_mode.get() == "log":
        IVax.set_yscale("log")
    else:
        IVax.set_yscale("linear")
    IVcanvas.draw()

def Reset():
    try:
        # Limpiar la figura IV (I-V Characteristic)
        IVfig.clf()  # Borra todos los ejes y trazos
        # (Opcional) Crear un nuevo eje vacío si lo necesitas
        IVax_new = IVfig.add_subplot(111)
        IVax.set_title("MOSFET I-V Characteristic", fontsize=10)
        IVax.set_xlabel("V_D [V]", fontsize=10)
        IVax.set_ylabel("I_D [A]", fontsize=10)
        IVax_new.grid(True, linestyle='--', linewidth=0.5, color="gray")
        IVcanvas.draw()  # Redibuja el canvas para reflejar el cambio
        I_D = None # Limpiamos I_D
        V_D = None # Limpiamos V_D
        V_G = None
    except Exception as e:
        print("The I-V figure couldn't be cleaned:", e)

    try:
        # Limpiar la figura VOVD (Output Voltage)
        VOVDfig.clf()  # Borra todos los ejes y trazos
        # (Opcional) Crear un nuevo eje vacío si lo necesitas
        VOVDax_new = VOVDfig.add_subplot(111)
        VOVDax.set_title("Output Voltage", fontsize=10)
        VOVDax.set_xlabel("V_O [V]", fontsize=10)
        VOVDax.set_ylabel("V_D [V]", fontsize=10)
        VOVDax_new.grid(True, linestyle='--', linewidth=0.5, color="gray")
        VOVDcanvas.draw()  # Redibuja el canvas para reflejar el cambio
        V_O = None # Limpiamos V_O
    except Exception as e:
        print("The V_O-V figure couldn't be cleaned:", e) 

def Save_IV():
    # Abre un diálogo para guardar el archivo
    filetypes = [("PNG Image", "*.png"),
                 ("JPEG Image", "*.jpg"),
                 ("SVG Image", "*.svg"),
                 ("PDF Document", "*.pdf"),
                 ("All Files", "*.*")]
    filename = filedialog.asksaveasfilename(
        title="Save Figure As",
        defaultextension=".png",
        filetypes=filetypes)
    
    if filename:
        try:
            IVfig.savefig(filename)
            messagebox.showinfo("Save Figure", f"Figure saved as:\n{filename}")
        except Exception as e:
            messagebox.showerror("Save Figure", f"Error saving figure:\n{e}")

def Save_VOVD():
    # Abre un diálogo para guardar el archivo
    filetypes = [("PNG Image", "*.png"),
                 ("JPEG Image", "*.jpg"),
                 ("SVG Image", "*.svg"),
                 ("PDF Document", "*.pdf"),
                 ("All Files", "*.*")]
    filename = filedialog.asksaveasfilename(
        title="Save Figure As",
        defaultextension=".png",
        filetypes=filetypes)
    
    if filename:
        try:
            VOVDfig.savefig(filename)
            messagebox.showinfo("Save Figure", f"Figure saved as:\n{filename}")
        except Exception as e:
            messagebox.showerror("Save Figure", f"Error saving figure:\n{e}")

def Save_config():
    # Crear un diccionario con los nombres y los valores actuales de cada Entry.
    data = {
        "VDmin": VDmin.get(),
        "VDmax": VDmax.get(),
        "VDsteps": VDsteps.get(),
        "VGmin": VGmin.get(),
        "VGmax": VGmax.get(),
        "VGsteps": VGsteps.get(),
        "R": R.get(),
        "NAveraging": NAveraging.get(),
        "IOff": IOff.get()
    }
    
    # Abrir un cuadro de diálogo "Guardar como..."
    filename = filedialog.asksaveasfilename(
        title="Save data in CSV",
        defaultextension=".csv",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if not filename:
        return  # Se canceló la selección

    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Escribir la primera fila con los nombres (cabeceras)
            headers = list(data.keys())
            writer.writerow(headers)
            # Escribir la segunda fila con los datos
            values = [data[key] for key in headers]
            writer.writerow(values)
        messagebox.showinfo("Saved", f"Data saved in:\n{filename}")
    except Exception as e:
        messagebox.showerror("Error", f"Error ocurred when saving file:\n{e}")

def Rename_IV():
    # Obtener las líneas (curvas) del eje
    lines = IVax.get_lines()
    if not lines:
        messagebox.showerror("Error", "There are no curves to rename.")
        return

    # Obtener la lista de etiquetas actuales de las curvas
    current_labels = [line.get_label() for line in lines]
    
    # Crear un mensaje que incluya las opciones disponibles
    prompt = ("Give the current name of the curve you want to rename.\n"
              "Options: " + ", ".join(current_labels))
    
    current_name = simpledialog.askstring("Select Curve", prompt)
    if current_name is None or current_name.strip() == "":
        messagebox.showerror("Error", "You must give a valid name.")
        return

    # Buscar la curva cuyo label coincide exactamente con el nombre ingresado
    selected_line = None
    for line in lines:
        if line.get_label() == current_name:
            selected_line = line
            break

    if selected_line is None:
        messagebox.showerror("Error", f"Couldn't find a curve named '{current_name}'.")
        return

    # Solicitar el nuevo nombre para la curva
    new_name = simpledialog.askstring("Rename Curve", "Give the new name for the curve:")
    if new_name is None or new_name.strip() == "":
        messagebox.showerror("Error", "You must give a valid name for the curve.")
        return

    # Actualizar la etiqueta de la curva seleccionada
    selected_line.set_label(new_name)
    
    # Actualizar la leyenda y redibujar el canvas
    IVax.legend()
    IVcanvas.draw()

def Rename_VOVD():
    # Obtener las líneas (curvas) del eje
    lines = VOVDax.get_lines()
    if not lines:
        messagebox.showerror("Error", "There are no curves to rename.")
        return

    # Obtener la lista de etiquetas actuales de las curvas
    current_labels = [line.get_label() for line in lines]
    
    # Crear un mensaje que incluya las opciones disponibles
    prompt = ("Give the current name of the curve you want to rename.\n"
              "Options: " + ", ".join(current_labels))
    
    current_name = simpledialog.askstring("Select Curve", prompt)
    if current_name is None or current_name.strip() == "":
        messagebox.showerror("Error", "You must give a valid name.")
        return

    # Buscar la curva cuyo label coincide exactamente con el nombre ingresado
    selected_line = None
    for line in lines:
        if line.get_label() == current_name:
            selected_line = line
            break

    if selected_line is None:
        messagebox.showerror("Error", f"Couldn't find a curve named '{current_name}'.")
        return

    # Solicitar el nuevo nombre para la curva
    new_name = simpledialog.askstring("Rename Curve", "Give the new name for the curve:")
    if new_name is None or new_name.strip() == "":
        messagebox.showerror("Error", "You must give a valid name for the curve.")
        return

    # Actualizar la etiqueta de la curva seleccionada
    selected_line.set_label(new_name)
    
    # Actualizar la leyenda y redibujar el canvas
    VOVDax.legend()
    VOVDcanvas.draw()

def Export_IV_Data():
    # Se asume que IVax es un objeto Axes de Matplotlib (por ejemplo, global)
    lines = IVax.get_lines()
    if not lines:
        messagebox.showerror("Error", "Couldn't find any data in MOSFET I-V Characteristic figure.")
        return

    data_dict = {}
    # Recorrer cada línea (curva)
    for idx, line in enumerate(lines):
        # Obtener la etiqueta de la curva; si no tiene etiqueta útil, se asigna un nombre por defecto.
        label = line.get_label()
        if not label or label.startswith('_'):
            label = f"Curve_{idx+1}"
        # Extraer los datos de x e y y convertirlos en un arreglo de NumPy
        x_data = np.array(line.get_xdata())
        y_data = np.array(line.get_ydata())
        # Guardar en el diccionario usando como claves el nombre de la curva + "_x" y "_y"
        data_dict[f"{label}_x"] = pd.Series(x_data)
        data_dict[f"{label}_y"] = pd.Series(y_data)

    # Crear un DataFrame con las Series; Pandas rellenará con NaN las filas faltantes
    df = pd.DataFrame(data_dict)

    # Abrir un diálogo para seleccionar dónde guardar el CSV
    filename = filedialog.asksaveasfilename(
        title="Save data as CSV",
        defaultextension=".csv",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if filename:
        try:
            df.to_csv(filename, index=False)
            messagebox.showinfo("Saved", f"Data saved in:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Error when saving file:\n{e}")

def Export_VOVD_Data():
    # Se asume que IVax es un objeto Axes de Matplotlib (por ejemplo, global)
    lines = VOVDax.get_lines()
    if not lines:
        messagebox.showerror("Error", "Couldn't find any data in MOSFET V_O-V_D Characteristic figure.")
        return

    data_dict = {}
    # Recorrer cada línea (curva)
    for idx, line in enumerate(lines):
        # Obtener la etiqueta de la curva; si no tiene etiqueta útil, se asigna un nombre por defecto.
        label = line.get_label()
        if not label or label.startswith('_'):
            label = f"Curve_{idx+1}"
        # Extraer los datos de x e y y convertirlos en un arreglo de NumPy
        x_data = np.array(line.get_xdata())
        y_data = np.array(line.get_ydata())
        # Guardar en el diccionario usando como claves el nombre de la curva + "_x" y "_y"
        data_dict[f"{label}_x"] = pd.Series(x_data)
        data_dict[f"{label}_y"] = pd.Series(y_data)

    # Crear un DataFrame con las Series; Pandas rellenará con NaN las filas faltantes
    df = pd.DataFrame(data_dict)

    # Abrir un diálogo para seleccionar dónde guardar el CSV
    filename = filedialog.asksaveasfilename(
        title="Save data as CSV",
        defaultextension=".csv",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if filename:
        try:
            df.to_csv(filename, index=False)
            messagebox.showinfo("Saved", f"Data saved in:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Error when saving file:\n{e}")

def Hide_IV():
    # Obtener las líneas (curvas) del eje
    lines = IVax.get_lines()
    if not lines:
        messagebox.showerror("Error", "There are no curves to hide.")
        return

    # Obtener los nombres de las curvas
    current_labels = [line.get_label() for line in lines]
    
    # Crear un mensaje que incluya las opciones disponibles
    prompt = ("Give the name of the curve you want to hide.\n"
              "Options: " + ", ".join(current_labels))
    
    curve_name = simpledialog.askstring("Select Curve", prompt)
    if curve_name is None or curve_name.strip() == "":
        messagebox.showerror("Error", "You must give a valid name.")
        return

    # Buscar la curva cuyo label coincide exactamente con el nombre ingresado
    selected_line = None
    for line in lines:
        if line.get_label() == curve_name:
            selected_line = line
            break

    if selected_line is None:
        messagebox.showerror("Error", f"Coulnd't find a curve named '{curve_name}'.")
        return

    # Ocultar la curva seleccionada
    selected_line.set_visible(False)

    # Actualizar la leyenda y redibujar el canvas
    IVax.legend()
    IVcanvas.draw()
    messagebox.showinfo("Curve has been hidden", f"The curve '{curve_name}' has been hidden.")

def Hide_VOVD():
    # Obtener las líneas (curvas) del eje
    lines = VOVDax.get_lines()
    if not lines:
        messagebox.showerror("Error", "There are no curves to hide.")
        return

    # Obtener los nombres de las curvas
    current_labels = [line.get_label() for line in lines]
    
    # Crear un mensaje que incluya las opciones disponibles
    prompt = ("Give the name of the curve you want to hide.\n"
              "Options: " + ", ".join(current_labels))
    
    curve_name = simpledialog.askstring("Select Curve", prompt)
    if curve_name is None or curve_name.strip() == "":
        messagebox.showerror("Error", "You must give a valid name.")
        return

    # Buscar la curva cuyo label coincide exactamente con el nombre ingresado
    selected_line = None
    for line in lines:
        if line.get_label() == curve_name:
            selected_line = line
            break

    if selected_line is None:
        messagebox.showerror("Error", f"Coulnd't find a curve named '{curve_name}'.")
        return

    # Ocultar la curva seleccionada
    selected_line.set_visible(False)

    # Actualizar la leyenda y redibujar el canvas
    VOVDax.legend()
    VOVDcanvas.draw()
    messagebox.showinfo("Curve has been hidden", f"The curve '{curve_name}' has been hidden.")

def Show_IV():
    # Obtener las líneas (curvas) del eje
    lines = IVax.get_lines()
    if not lines:
        messagebox.showerror("Error", "There are no curves to show.")
        return

    # Obtener los nombres de las curvas
    current_labels = [line.get_label() for line in lines]
    
    # Crear un mensaje que incluya las opciones disponibles
    prompt = ("Give the name of the curve you want to show.\n"
              "Options: " + ", ".join(current_labels))
    
    curve_name = simpledialog.askstring("Select Curve", prompt)
    if curve_name is None or curve_name.strip() == "":
        messagebox.showerror("Error", "You must give a valid name.")
        return

    # Buscar la curva cuyo label coincide exactamente con el nombre ingresado
    selected_line = None
    for line in lines:
        if line.get_label() == curve_name:
            selected_line = line
            break

    if selected_line is None:
        messagebox.showerror("Error", f"Coulnd't find a curve named '{curve_name}'.")
        return

    # Ocultar la curva seleccionada
    selected_line.set_visible(True)

    # Actualizar la leyenda y redibujar el canvas
    IVax.legend()
    IVcanvas.draw()
    messagebox.showinfo("Curve is shown", f"The curve '{curve_name}' is shown.")

def Show_VOVD():
    # Obtener las líneas (curvas) del eje
    lines = VOVDax.get_lines()
    if not lines:
        messagebox.showerror("Error", "There are no curves to show.")
        return

    # Obtener los nombres de las curvas
    current_labels = [line.get_label() for line in lines]
    
    # Crear un mensaje que incluya las opciones disponibles
    prompt = ("Give the name of the curve you want to show.\n"
              "Options: " + ", ".join(current_labels))
    
    curve_name = simpledialog.askstring("Select Curve", prompt)
    if curve_name is None or curve_name.strip() == "":
        messagebox.showerror("Error", "You must give a valid name.")
        return

    # Buscar la curva cuyo label coincide exactamente con el nombre ingresado
    selected_line = None
    for line in lines:
        if line.get_label() == curve_name:
            selected_line = line
            break

    if selected_line is None:
        messagebox.showerror("Error", f"Coulnd't find a curve named '{curve_name}'.")
        return

    # Ocultar la curva seleccionada
    selected_line.set_visible(True)

    # Actualizar la leyenda y redibujar el canvas
    VOVDax.legend()
    VOVDcanvas.draw()
    messagebox.showinfo("Curve is shown", f"The curve '{curve_name}' is shown.")

def Load_IV():
    # Abrir un diálogo para seleccionar el archivo CSV
    filename = filedialog.askopenfilename(
        title="Choose CSV File",
        defaultextension=".csv",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if not filename:
        return  # Si se cancela la selección, no hacer nada

    try:
        # Leer los datos desde el archivo CSV
        df = pd.read_csv(filename)
    except Exception as e:
        messagebox.showerror("Error", f"Error encountered when reading CSV file:\n{e}")
        return

    # Verificar que las columnas sean pares (x, y) de acuerdo con el formato
    if len(df.columns) % 2 != 0:
        messagebox.showerror("Error", "CSV file doesn't contain a valid number of rows (must be even).")
        return

    # Recorrer las columnas del DataFrame para trazar las curvas
    for i in range(0, len(df.columns), 2):
        x_data = df.iloc[:, i]    # Columnas impares (x)
        y_data = df.iloc[:, i+1]  # Columnas pares (y)

        # Obtener el nombre de la curva desde la primera fila (nombre_x, nombre_y)
        curve_name = df.columns[i].replace('_x', '').replace('_y', '')  # Eliminar "_x" o "_y" del nombre

        try:
            # Graficar la curva
            IVax.plot(x_data, y_data, label=curve_name)
        except AttributeError:
            messagebox.showerror('Error', "In order to load data, run a measurement to create a figure. Right now, the axis don't exist, so the program doesn't know where to represent the data.")
            return

    # Actualizar la leyenda y redibujar el canvas
    IVax.legend()
    IVcanvas.draw()
    messagebox.showinfo("Data Loaded", f"Data loaded from: {filename}")

def Load_VOVD():
    # Abrir un diálogo para seleccionar el archivo CSV
    filename = filedialog.askopenfilename(
        title="Choose CSV File",
        defaultextension=".csv",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if not filename:
        return  # Si se cancela la selección, no hacer nada

    try:
        # Leer los datos desde el archivo CSV
        df = pd.read_csv(filename)
    except Exception as e:
        messagebox.showerror("Error", f"Error encountered when reading CSV file:\n{e}")
        return

    # Verificar que las columnas sean pares (x, y) de acuerdo con el formato
    if len(df.columns) % 2 != 0:
        messagebox.showerror("Error", "CSV file doesn't contain a valid number of rows (must be even).")
        return

    # Recorrer las columnas del DataFrame para trazar las curvas
    for i in range(0, len(df.columns), 2):
        x_data = df.iloc[:, i]    # Columnas impares (x)
        y_data = df.iloc[:, i+1]  # Columnas pares (y)

        # Obtener el nombre de la curva desde la primera fila (nombre_x, nombre_y)
        curve_name = df.columns[i].replace('_x', '').replace('_y', '')  # Eliminar "_x" o "_y" del nombre

        try:
            # Graficar la curva
            VOVDax.plot(x_data, y_data, label=curve_name)
        except AttributeError:
            messagebox.showerror('Error', "In order to load data, run a measurement to create a figure. Right now, the axes don't exist, so the program doesn't know where to represent the data.")
            return

    # Actualizar la leyenda y redibujar el canvas
    VOVDax.legend()
    VOVDcanvas.draw()
    messagebox.showinfo("Data Loaded", f"Data loaded from: {filename}")

def Fullscalerange():
    if V_O is None:
        messagebox.showerror("Error", "Represent a curve in order to show the FSR.")
        return
    vomax = np.max(V_O)
    vomin = np.min(V_O)
    fsr = vomax - vomin
    fsrpercent = fsr/20*100
    messagebox.showinfo("Full Scale Range (FSR)", f"The FSR is: {fsr: .2f} V, or {fsrpercent: .2f} %")

def StrInv():
    global StrInv_window, StrInv2_window, Vd

    if sweep_val.get():
        messagebox.showerror('Error', 'Extra Parameters is not available for sweep data.')
        return
    else:

        if XAxis_value.get() == 'V_GS':
            def Calculate():
                def mosfet_model(Vg, UO, THETA, VTO):
                    return UO / (1 + THETA * (Vg - VTO)) * 3.4530e-6 * (Vg - VTO) * Vd

                if V_O is None or I_D is None:
                    messagebox.showerror('Error', 'Run a measurement in order to obtain the parameters.')
                    return

                # Supongamos que tus datos están en arrays numpy llamados I_D_mean, V_O_mean, V_D, V_G
                valid_V = V_O < 10  # Condición de filtrado
                xData = V_G[valid_V]  # Solo los valores de V_G que cumplen la condición
                Vd = V_D[valid_V]  # Solo los valores de V_D que cumplen la condición
                Vd = np.mean(Vd)  # Promedio de V_D
                yData = I_D[valid_V]  # Solo los valores de I_D_mean que cumplen la condición
                valid_x = xData > 1.2
                xData = xData[valid_x]
                yData = yData[valid_x]

                # Definir los parámetros iniciales y los límites
                params_initial = [500, 0.2, 1.5]  # Valores iniciales para UO, THETA, VTO
                params_bounds = ([0, 0, 0], [np.inf, np.inf, 5])  # Límites inferiores y superiores para los parámetros
                
                # Definir la función de ajuste
                popt, pcov = curve_fit(mosfet_model, xData, yData, p0=params_initial, bounds=params_bounds)
                
                # popt contiene los parámetros ajustados: [UO, THETA, VTO]
                UO_fit, THETA_fit, VTO_fit = popt
                
                # Estimar los errores de los parámetros (desviación estándar de cada parámetro)
                errors = np.sqrt(np.diag(pcov))
                error_UO = errors[0]
                error_THETA = errors[1]
                error_VTO = errors[2]
                
                # Calcular la predicción del modelo ajustado
                I_D_pred = mosfet_model(xData, *popt)
                
                # Calcular R² (coeficiente de determinación)
                SS_res = np.sum((yData - I_D_pred)**2)
                SS_tot = np.sum((yData - np.mean(yData))**2)
                R2 = 1 - (SS_res / SS_tot)

                # Asignar resultados a los campos de la interfaz
                UO_val.insert(0, f"{UO_fit: .7g}")
                THETA_val.insert(0, f"{THETA_fit: .7g}")
                VTO_val.insert(0, f"{VTO_fit: .7g}")
                R2_val.insert(0, f"{R2: .7g}")
                errorUO.insert(0, f"{error_UO: .7g}")
                errorTHETA.insert(0, f"{error_THETA: .7g}")
                errorVTO.insert(0, f"{error_VTO: .7g}")

                # La intensidad que representaremos será
                I_D_degrad = mosfet_model(xData, UO_fit, THETA_fit, VTO_fit)

                # Puntos para la curva VO
                V_O_degrad = R_val * I_D_degrad + Vd

                IVax.plot(xData, I_D_degrad, label="Degradation fit", linewidth=1)
                IVax.legend()
                IVcanvas.draw()
                VOVDax.plot(xData, V_O_degrad, label='Degradation fit', linewidth=1)
                VOVDax.legend()
                VOVDcanvas.draw()

            def Export_StrInv_Data():
                # Crear un diccionario con los nombres y los valores actuales de cada Entry.
                data = {
                    "UO": UO_val.get(),
                    "THETA": THETA_val.get(),
                    "VTO": VTO_val.get(),
                    "ErrorUO": errorUO.get(),
                    "ErrorTHETA": errorTHETA.get(),
                    "ErrorVTO": errorVTO.get(),
                    "R2": R2_val.get(),
                }
                
                # Abrir un cuadro de diálogo "Guardar como..."
                filename = filedialog.asksaveasfilename(
                    title="Save data in CSV",
                    defaultextension=".csv",
                    filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
                )
                if not filename:
                    return  # Se canceló la selección

                try:
                    with open(filename, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        # Escribir la primera fila con los nombres (cabeceras)
                        headers = list(data.keys())
                        writer.writerow(headers)
                        # Escribir la segunda fila con los datos
                        values = [data[key] for key in headers]
                        writer.writerow(values)
                    messagebox.showinfo("Saved", f"Data saved in:\n{filename}")
                except Exception as e:
                    messagebox.showerror("Error", f"Error ocurred when saving file:\n{e}")
                
            if StrInv_window and StrInv_window.winfo_exists():
                StrInv_window.lift()
                StrInv_window.focus_force()  # Trae la ventana al frente y la pone en foco
            else:
                StrInv_window = Toplevel(root)
                StrInv_window.title("Extra Parameters")
                StrInv_window.iconbitmap('US-marca-principal.ico')
                StrInv_window.geometry("600x290")
                StrInv_window.resizable(False, False)

                # Crear un menú bar
                menubar = Menu(StrInv_window)

                # Crear el menú "File" y sus opciones
                file_menu = Menu(menubar, tearoff=0)
                # Crear el submenú para 'Save Figure'
                file_menu.add_command(label="Export Parameters", command=Export_StrInv_Data)

                # Agregar los menús al menú bar
                menubar.add_cascade(label="Options", menu=file_menu)

                # Configurar el menú de la ventana principal
                StrInv_window.config(menu=menubar)
                
                OhmicSat_label = Label(StrInv_window, text="OHMIC AND SATURATION PARAMETERS", padx=10, pady=5)
                OhmicSat_label.grid(row=0, column=0, columnspan=10)

                # Cargar y redimensionar la ecuación de la intensidad que atraviesa el diodo
                StrInvEquation = Image.open("MOSFETCharacteristicsUOTHETAVTO.png")  # Cambia por la ruta de tu imagen
                StrInvEquation_resized = StrInvEquation.resize((300, 30))  # Cambia el tamaño (ancho, alto)
                StrInvEquation_photo = ImageTk.PhotoImage(StrInvEquation_resized)
                StrInv_window.StrInvEquation_photo = StrInvEquation_photo
                StrInvEquation_label = Label(StrInv_window, image=StrInvEquation_photo)
                StrInvEquation_label.grid(row=1, columnspan=10)  # Cambia la posición (x, y)

                Calculate = Button(StrInv_window, text="Calculate", command=Calculate)
                Calculate.grid(row=2, column=0, columnspan=10)

                # Create label for Vth
                UO_label = Label(StrInv_window, text="UO (cm^2/(V·s))", padx=10, pady=10)
                UO_label.grid(row=3, column=0, sticky="e")

                # Create entry for Vth
                UO_val = Entry(StrInv_window, width=15)
                UO_val.grid(row=3, column=1, sticky="w")

                # Create label for errorVth
                errorUO_label = Label(StrInv_window, text="Error UO (cm^2/(V·s))", padx=10, pady=10)
                errorUO_label.grid(row=3, column=5, sticky="e")

                # Create entry for errorVth
                errorUO = Entry(StrInv_window, width=15)
                errorUO.grid(row=3, column=6, sticky="w")

                # Create label for Vth
                THETA_label = Label(StrInv_window, text="THETA (1/V)", padx=10, pady=10)
                THETA_label.grid(row=4, column=0, sticky="e")

                # Create entry for Vth
                THETA_val = Entry(StrInv_window, width=15)
                THETA_val.grid(row=4, column=1, sticky="w")

                # Create label for errorVth
                errorTHETA_label = Label(StrInv_window, text="Error THETA (1/V)", padx=10, pady=10)
                errorTHETA_label.grid(row=4, column=5, sticky="e")

                # Create entry for errorVth
                errorTHETA = Entry(StrInv_window, width=15)
                errorTHETA.grid(row=4, column=6, sticky="w")

                # Create label for Vth
                VTO_label = Label(StrInv_window, text="VTO (V)", padx=10, pady=10)
                VTO_label.grid(row=5, column=0, sticky="e")

                # Create entry for Vth
                VTO_val = Entry(StrInv_window, width=15)
                VTO_val.grid(row=5, column=1, sticky="w")

                # Create label for errorVth
                errorVTO_label = Label(StrInv_window, text="Error VTO (V)", padx=10, pady=10)
                errorVTO_label.grid(row=5, column=5, sticky="e")

                # Create entry for errorVth
                errorVTO = Entry(StrInv_window, width=15)
                errorVTO.grid(row=5, column=6, sticky="w")

                # Create label for Vth
                R2_label = Label(StrInv_window, text="R^2", padx=10, pady=10)
                R2_label.grid(row=6, column=3, sticky="e")

                # Create entry for Vth
                R2_val = Entry(StrInv_window, width=15)
                R2_val.grid(row=6, column=4, sticky="w")

        else:

            def Calculate():

                def fopt2(p, uo, theta, vto, d, vsb):
                    """
                    Calcula fopt2 para un vector p = [GAMMA, PHI, KAPPA, THETA2].
                    Devuelve un vector f = id - idx.
                    
                    Parámetros:
                    p : array_like, [GAMMA, PHI, KAPPA, THETA2]
                    uo, theta, vto : parámetros fijos (obtenidos del ajuste previo)
                    d  : array de datos, con d[:,0] = vds y d[:,1] = idx
                    vsb : voltaje de substrato

                    Retorna:
                    f : diferencia entre la corriente calculada (id) y la medida (idx)
                    """
                    # Extraemos los parámetros a ajustar
                    GAMMA = p[0]
                    PHI = p[1]
                    KAPPA = p[2]
                    THETA2 = p[3]
                    
                    # Datos experimentales
                    vds = d[:, 0]
                    idx = d[:, 1]
                    
                    # Parámetros conocidos (se definen fijos en el modelo)
                    vgs = 4
                    eox = 3.453e-13
                    esi = 1.03592e-12
                    tox = 100e-9
                    w_l = 100
                    q = 1.602e-19
                    nsub = 2.74e16
                    Leff = 5e-6
                    
                    # Cálculos
                    fb = GAMMA / (4 * np.sqrt(PHI + vsb))         # efecto substrato
                    vt = vto + GAMMA * (np.sqrt(vsb + PHI) - np.sqrt(PHI))  # voltaje umbral ajustado
                    ueff = uo / (1 + theta * (vgs - vt))           # movilidad efectiva
                    beta = ueff * eox / tox * w_l                  # factor beta
                    vdsat = (vgs - vt) / (1 + fb)                  # voltaje de saturación
                    vde = np.minimum(vds, vdsat)                   # toma el mínimo (para la región lineal)
                    id_calc = beta * (vgs - vt - (1 + fb) / 2 * vde) * vde  # cálculo de ID en triodo
                    
                    # Para los puntos en saturación (vds > vdsat)
                    sat = np.where(vds > vdsat)[0]
                    if sat.size > 0:
                        deltaL = np.sqrt(2 * esi / (q * nsub)) * np.sqrt(KAPPA * (vds[sat] - vdsat))
                        id_calc[sat] = id_calc[sat] / (1 - deltaL / Leff)
                    
                    # f es la diferencia entre el modelo calculado y el dato experimental
                    f = id_calc - idx
                    return f

                messagebox.showinfo('Information', 'This functionality is in development. Some parameters (GAMMA, PHI, KAPPA and THETA2) might not be accurately calculated.')

                if V_O is None or I_D is None:
                    messagebox.showerror('Error', 'Run a measurement in order to obtain the parameters.')
                    return

                # Supongamos que tus datos están en arrays numpy llamados I_D_mean, V_O_mean, V_D, V_G
                valid_V = V_O < 10  # Condición de filtrado
                xData = V_G[valid_V]  # Solo los valores de V_G que cumplen la condición
                Vd = V_D[valid_V]  # Solo los valores de V_D que cumplen la condición
                Vd = np.mean(Vd)  # Promedio de V_D
                yData = I_D[valid_V]  # Solo los valores de I_D_mean que cumplen la condición
                valid_x = xData > 1.2
                xData = xData[valid_x]
                yData = yData[valid_x]
                Vsb_val = float(VSB_val.get())
                UO = float(UO2_val.get())
                THETA = float(THETA_2_val.get())
                VTO = float(VTO2_val.get())

                # --- Ajuste de p2 usando least_squares ---

                # Supón que ya tienes:
                par_initial = np.array([0.8, 0.7, 1.0, 0.0])
                par_min = np.array([0, 0, -10.0, 0])
                par_max = np.array([10, 10, 10, 2])

                # Opciones de tolerancia (equivalentes a TolX y TolFun)
                # Aquí se definen xtol y ftol; los valores pueden ajustarse
                #xtol = 5e-16
                #ftol = 5e-16

                # Supón que d se construye concatenando xData y yData, 
                # por ejemplo:
                #   xData -> vds (vector) y yData -> idx (vector)
                d = np.column_stack([xData, yData])

                # Ahora, realizar el ajuste con least_squares:
                res = least_squares(lambda p: fopt2(p, UO, THETA, VTO, d, Vsb_val),
                                    par_initial, bounds=(par_min, par_max))

                # Extraer los parámetros ajustados:
                p2 = res.x  # p2[0] = gamma, p2[1] = phi, p2[2] = kappa, p2[3] = theta2
                GAMMA_fit = p2[0]
                PHI_fit = p2[1]
                KAPPA_fit = p2[2]
                THETA2_fit = p2[3]

                GAMMA_val.insert(0, f"{GAMMA_fit: .7g}")
                PHI_val.insert(0, f"{PHI_fit: .7g}")
                KAPPA_val.insert(0, f"{KAPPA_fit: .7g}")
                THETA2_val.insert(0, f"{THETA2_fit: .7g}")
            
            def Export_StrInv2_Data():
                # Crear un diccionario con los nombres y los valores actuales de cada Entry.
                data = {
                    "GAMMA": GAMMA_val.get(),
                    "PHI": PHI_val.get(),
                    "KAPPA": KAPPA_val.get(),
                    "THETA2": THETA_2_val.get(),
                    "VSB": VSB_val.get(),
                }
                
                # Abrir un cuadro de diálogo "Guardar como..."
                filename = filedialog.asksaveasfilename(
                    title="Save data in CSV",
                    defaultextension=".csv",
                    filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
                )
                if not filename:
                    return  # Se canceló la selección

                try:
                    with open(filename, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        # Escribir la primera fila con los nombres (cabeceras)
                        headers = list(data.keys())
                        writer.writerow(headers)
                        # Escribir la segunda fila con los datos
                        values = [data[key] for key in headers]
                        writer.writerow(values)
                    messagebox.showinfo("Saved", f"Data saved in:\n{filename}")
                except Exception as e:
                    messagebox.showerror("Error", f"Error ocurred when saving file:\n{e}")

            if StrInv2_window and StrInv2_window.winfo_exists():
                StrInv2_window.lift()
                StrInv2_window.focus_force()  # Trae la ventana al frente y la pone en foco
            else:
                StrInv2_window = Toplevel(root)
                StrInv2_window.title("Extra Parameters")
                StrInv2_window.iconbitmap('US-marca-principal.ico')
                StrInv2_window.geometry("635x360")
                StrInv2_window.resizable(False, False)

                # Crear un menú bar
                menubar = Menu(StrInv2_window)

                # Crear el menú "File" y sus opciones
                file_menu = Menu(menubar, tearoff=0)
                # Crear el submenú para 'Save Figure'
                file_menu.add_command(label="Export Parameters", command=Export_StrInv2_Data)

                # Agregar los menús al menú bar
                menubar.add_cascade(label="Options", menu=file_menu)

                # Configurar el menú de la ventana principal
                StrInv2_window.config(menu=menubar)

                StrInv_label = Label(StrInv2_window, text="STRONG INVERSION PARAMETERS", padx=10, pady=5)
                StrInv_label.grid(row=0, column=0, columnspan=10)

                # Creamos botón para Calculate
                Calculate = Button(StrInv2_window, text='Calculate', command=Calculate)
                Calculate.grid(row=1, columnspan=10)

                Substrate_label = Label(StrInv2_window, text="Is there substrate effect?", padx=10, pady=5)
                Substrate_label.grid(row=2, column=0, columnspan=10)

                # Create label for Vth
                VSB_label = Label(StrInv2_window, text="VSB (V)", padx=10, pady=10)
                VSB_label.grid(row=3, column=3, sticky="e")

                # Create entry for Vth
                VSBdefault = IntVar()
                VSBdefault.set(0)
                VSB_val = Entry(StrInv2_window, width=15, textvariable=VSBdefault)
                VSB_val.grid(row=3, column=4, sticky="w")

                # Create label for Vth
                UO2_label = Label(StrInv2_window, text="UO (cm^2/(V·s))", padx=10, pady=10)
                UO2_label.grid(row=4, column=3, sticky="e")

                # Create entry for Vth
                UO2default = IntVar()
                UO2default.set(0)
                UO2_val = Entry(StrInv2_window, width=15, textvariable=UO2default)
                UO2_val.grid(row=4, column=4, sticky="w")

                # Create label for Vth
                THETA_2_label = Label(StrInv2_window, text="THETA (1/V)", padx=10, pady=10)
                THETA_2_label.grid(row=5, column=3, sticky="e")

                # Create entry for Vth
                THETA_2default = IntVar()
                THETA_2default.set(0)
                THETA_2_val = Entry(StrInv2_window, width=15, textvariable=THETA_2default)
                THETA_2_val.grid(row=5, column=4, sticky="w")

                # Create label for Vth
                VTO2_label = Label(StrInv2_window, text="VTO (V)", padx=10, pady=10)
                VTO2_label.grid(row=6, column=3, sticky="e")

                # Create entry for Vth
                VTO2default = IntVar()
                VTO2default.set(0)
                VTO2_val = Entry(StrInv2_window, width=15, textvariable=VTO2default)
                VTO2_val.grid(row=6, column=4, sticky="w")

                # Create label for Vth
                GAMMA_label = Label(StrInv2_window, text="GAMMA (V^(1/2))", padx=10, pady=10)
                GAMMA_label.grid(row=7, column=0, sticky="e")

                # Create entry for Vth
                GAMMA_val = Entry(StrInv2_window, width=15)
                GAMMA_val.grid(row=7, column=1, sticky="w")

                # Create label for errorVth
                KAPPA_label = Label(StrInv2_window, text="KAPPA (1/V)", padx=10, pady=10)
                KAPPA_label.grid(row=7, column=5, sticky="e")

                # Create entry for errorVth
                KAPPA_val = Entry(StrInv2_window, width=15)
                KAPPA_val.grid(row=7, column=6, sticky="w")

                # Create label for Vth
                PHI_label = Label(StrInv2_window, text="PHI (V)", padx=10, pady=10)
                PHI_label.grid(row=8, column=0, sticky="e")

                # Create entry for Vth
                PHI_val = Entry(StrInv2_window, width=15)
                PHI_val.grid(row=8, column=1, sticky="w")

                # Create label for errorVth
                THETA2_label = Label(StrInv2_window, text="THETA2 (1/V)", padx=10, pady=10)
                THETA2_label.grid(row=8, column=5, sticky="e")

                # Create entry for errorVth
                THETA2_val = Entry(StrInv2_window, width=15)
                THETA2_val.grid(row=8, column=6, sticky="w")

root = Tk()
root.title("MOSFET Characteristics Application")
root.iconbitmap('US-marca-principal.ico')
root.geometry("1100x700")
root.resizable(False, False)

# Crear un menú bar
menubar = Menu(root)

# Crear el menú "File" y sus opciones
file_menu = Menu(menubar, tearoff=0)
# Crear el submenú para 'Save Figure'
save_menu = Menu(menubar, tearoff=0)
save_menu.add_command(label="Save I-V MOSFET Characteristic", command=Save_IV)
save_menu.add_command(label="Save V-V MOSFET Characteristic", command=Save_VOVD)
file_menu.add_cascade(label="Save Figure...", menu=save_menu)
file_menu.add_command(label="Save Config", command=Save_config)
rename_menu = Menu(menubar, tearoff=0)
rename_menu.add_command(label="Rename in I-V MOSFET Characteristic", command=Rename_IV)
rename_menu.add_command(label="Rename in V-V MOSFET Characteristic", command=Rename_VOVD)
file_menu.add_cascade(label="Rename Curve...", menu=rename_menu)
export_menu = Menu(menubar, tearoff=0)
export_menu.add_command(label="Export I-V MOSFET Characteristic Data", command=Export_IV_Data)
export_menu.add_command(label="Export V-V MOSFET Characteristic Data", command=Export_VOVD_Data)
file_menu.add_cascade(label="Export Data...", menu=export_menu)
hide_menu = Menu(menubar, tearoff=0)
hide_menu.add_command(label="Hide I-V MOSFET Characteristic Curve", command=Hide_IV)
hide_menu.add_command(label="Hide V-V MOSFET Characteristic Curve", command=Hide_VOVD)
file_menu.add_cascade(label="Hide Curve...", menu=hide_menu)
show_menu = Menu(menubar, tearoff=0)
show_menu.add_command(label="Show I-V MOSFET Characteristic Curve", command=Show_IV)
show_menu.add_command(label="Show V-V MOSFET Characteristic Curve", command=Show_VOVD)
file_menu.add_cascade(label="Show Curve...", menu=show_menu)
load_menu = Menu(menubar, tearoff=0)
load_menu.add_command(label="Load I-V MOSFET Characteristic Curve", command=Load_IV)
load_menu.add_command(label="Load V-V MOSFET Characteristic Curve", command=Load_VOVD)
file_menu.add_cascade(label="Load Curve...", menu=load_menu)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

# Crear el menú "Help" y sus opciones
help_menu = Menu(menubar, tearoff=0)
help_menu.add_command(label="Help", command=Open_help)
help_menu.add_command(label="Documentation", command=Open_documentation)

# Agregar los menús al menú bar
menubar.add_cascade(label="File", menu=file_menu)
menubar.add_cascade(label="Help", menu=help_menu)

# Configurar el menú de la ventana principal
root.config(menu=menubar)

# Create the NI myDAQ panel
NImyDAQ = LabelFrame(root, text="NI myDAQ", width=175, height=141, padx=10, pady=10)
NImyDAQ.place(x=29, y=700-531-141)
NImyDAQ.grid_propagate(False)

# Configurar las columnas para que se expandan proporcionalmente
for col in range(3):  # Total de columnas: 0 a 4
    NImyDAQ.grid_columnconfigure(col, weight=1)

# Configurar las filas para que se expandan proporcionalmente
for row in range(4):  # Total de filas: 0 a 3
    NImyDAQ.grid_rowconfigure(row, weight=1)

# Create label for Device ID
DeviceID_label = Label(NImyDAQ, text="Device ID", padx=10, pady=5)
DeviceID_label.grid(row=0, column=0, sticky="e")

# Create the Device ID Edit Field
DeviceIDdefault = StringVar()
DeviceIDdefault.set("myDAQ1")
DeviceID = Entry(NImyDAQ, width=10, textvariable=DeviceIDdefault)
DeviceID.grid(row=0, column=1, sticky="w")

# Create label that changes color
CheckNImyDAQ_label = Label(NImyDAQ, text="                                  ", bg="red", padx=10, pady=5)
CheckNImyDAQ_label.config(bg="red")
CheckNImyDAQ_label.grid(row=2, column=0, columnspan=2)

# Create button for Check NI myDAQ
CheckNImyDAQ = Button(NImyDAQ, text="Check NI myDAQ", padx=10, pady=5, command=ChechNImyDAQ)
CheckNImyDAQ.grid(row=1, column=0, columnspan=2)

# Create the Input Signal Properties panel
InputSignalProperties = LabelFrame(root, text="Input Signal Properties", width=410, height=141, padx=10, pady=10)
InputSignalProperties.place(x=246, y=700-531-141)
InputSignalProperties.grid_propagate(False)

# Configurar las columnas para que se expandan proporcionalmente
for col in range(5):  # Total de columnas: 0 a 4
    InputSignalProperties.grid_columnconfigure(col, weight=1)

# Configurar las filas para que se expandan proporcionalmente
for row in range(4):  # Total de filas: 0 a 3
    InputSignalProperties.grid_rowconfigure(row, weight=1)

# Create label for VDmin
VDmin_label = Label(InputSignalProperties, text="VDmin (V)", padx=10, pady=5)
VDmin_label.grid(row=0, column=0, sticky="e")  # Alinear a la derecha

# Create the VDmin Edit Field
VDmindefault = IntVar()
VDmindefault.set(0)
VDmin = Entry(InputSignalProperties, width=15, textvariable=VDmindefault)
VDmin.grid(row=0, column=1, sticky="w")  # Alinear a la izquierda

# Create label for VDmax
VDmax_label = Label(InputSignalProperties, text="VDmax (V)", padx=10, pady=5)
VDmax_label.grid(row=2, column=0, sticky="e")  # Alinear a la derecha

# Create the VDmax Edit Field
VDmaxdefault = IntVar()
VDmaxdefault.set(5)
VDmax = Entry(InputSignalProperties, width=15, textvariable=VDmaxdefault)
VDmax.grid(row=2, column=1, sticky="w")  # Alinear a la izquierda

# Create label for Vsteps
VDsteps_label = Label(InputSignalProperties, text="VDsteps", padx=10, pady=5)
VDsteps_label.grid(row=3, column=0, sticky="e")  # Alinear a la derecha

# Create the Vsteps Edit Field
VDstepsdefault = IntVar()
VDstepsdefault.set(25)
VDsteps = Entry(InputSignalProperties, width=15, textvariable=VDstepsdefault)
VDsteps.grid(row=3, column=1, sticky="w")  # Alinear a la izquierda

# Create label for VGmin
VGmin_label = Label(InputSignalProperties, text="VGmin (V)", padx=10, pady=5)
VGmin_label.grid(row=0, column=3, sticky="e")  # Alinear a la derecha

# Create the VGmin Edit Field
VGmindefault = IntVar()
VGmindefault.set(2.7)
VGmin = Entry(InputSignalProperties, width=15, textvariable=VGmindefault)
VGmin.grid(row=0, column=4, sticky="w")  # Alinear a la izquierda

# Create label for VGmax
VGmax_label = Label(InputSignalProperties, text="VGmax (V)", padx=10, pady=5)
VGmax_label.grid(row=2, column=3, sticky="e")  # Alinear a la derecha

# Create the VGmax Edit Field
VGmaxdefault = IntVar()
VGmaxdefault.set(2.7)
VGmax = Entry(InputSignalProperties, width=15, textvariable=VGmaxdefault)
VGmax.grid(row=2, column=4, sticky="w")  # Alinear a la izquierda

# Create label for VGsteps
VGsteps_label = Label(InputSignalProperties, text="VGsteps", padx=10, pady=5)
VGsteps_label.grid(row=3, column=3, sticky="e")  # Alinear a la derecha

# Create the VGsteps Edit Field
VGstepsdefault = IntVar()
VGstepsdefault.set(25)
VGsteps = Entry(InputSignalProperties, width=15, textvariable=VGstepsdefault)
VGsteps.grid(row=3, column=4, sticky="w")  # Alinear a la izquierda

# Create checkbox for Sweep
sweep_val = BooleanVar(value=False)
Sweep = Checkbutton(InputSignalProperties, text='Sweep', variable=sweep_val)
Sweep.grid(row=4, columnspan=10)

# Create the Measurement Properties panel
MeasurementProperties = LabelFrame(root, text="Measurement Properties", width=395, height=141, padx=10, pady=10)
MeasurementProperties.place(x=675, y=700-531-141)
MeasurementProperties.grid_propagate(False)

# Configurar las columnas para que se expandan proporcionalmente
for col in range(3):  # Total de columnas: 0 a 4
    MeasurementProperties.grid_columnconfigure(col, weight=1)

# Configurar las filas para que se expandan proporcionalmente
for row in range(4):  # Total de filas: 0 a 3
    MeasurementProperties.grid_rowconfigure(row, weight=1)

# Create label for R
R_label = Label(MeasurementProperties, text="R (Ohm)", padx=10, pady=5)
R_label.grid(row=0, column=0, sticky="e")

# Create the R Edit Field
Rdefault = IntVar()
Rdefault.set(1000)
R = Entry(MeasurementProperties, width=15, textvariable=Rdefault)
R.grid(row=0, column=1, sticky="w")

# Create label for NAveraging
NAveraging_label = Label(MeasurementProperties, text="NAveraging", padx=10, pady=5)
NAveraging_label.grid(row=1, column=0, sticky="e")

# Create the NAveraging Edit Field
NAveragingdefault = IntVar()
NAveragingdefault.set(1)
NAveraging = Entry(MeasurementProperties, width=15, textvariable=NAveragingdefault)
NAveraging.grid(row=1, column=1, sticky="w")

# Create label for IOff
IOff_label = Label(MeasurementProperties, text="IOff (A)", padx=10, pady=5)
IOff_label.grid(row=2, column=0, sticky="e")

# Create the IOff Edit Field
IOffdefault = IntVar()
IOffdefault.set(0)
IOff = Entry(MeasurementProperties, width=15, textvariable=IOffdefault)
IOff.grid(row=2, column=1, sticky="w")

# Create the NI myDAQ Ports panel
NImyDAQPorts = LabelFrame(root, text="NI myDAQ Ports", width=175, height=139, padx=10, pady=10)
NImyDAQPorts.place(x=28, y=700-375-139)
NImyDAQPorts.grid_propagate(False)

# Configurar las columnas para que se expandan proporcionalmente
for col in range(3):  # Total de columnas: 0 a 4
    NImyDAQPorts.grid_columnconfigure(col, weight=1)

# Configurar las filas para que se expandan proporcionalmente
for row in range(4):  # Total de filas: 0 a 3
    NImyDAQPorts.grid_rowconfigure(row, weight=1)

# Create label for VD
VD_label = Label(NImyDAQPorts, text="VD", padx=10, pady=1)
VD_label.grid(row=0, column=0, sticky="e")

# Create the VD DropDown
VD_options = ['ao0', 'ao1']
VD_value = StringVar(value=VD_options[0])
VD = OptionMenu(NImyDAQPorts, VD_value, *VD_options)
VD.grid(row=0, column=1, padx=10, pady=1, sticky="w")

# Create label for VG
VG_label = Label(NImyDAQPorts, text="VG", padx=10, pady=1)
VG_label.grid(row=1, column=0, sticky="e")

# Create the VG DropDown
VG_options = ['ao0', 'ao1']
VG_value = StringVar(value=VG_options[1])
VG = OptionMenu(NImyDAQPorts, VG_value, *VG_options)
VG.grid(row=1, column=1, padx=10, pady=1, sticky="w")

# Create label for VO
VO_label = Label(NImyDAQPorts, text="VO", padx=10, pady=1)
VO_label.grid(row=3, column=0, sticky="e")

# Create the VG DropDown
VO_options = ['ai0', 'ai1']
VO_value = StringVar(value=VO_options[1])
VO = OptionMenu(NImyDAQPorts, VO_value, *VO_options)
VO.grid(row=3, column=1, padx=10, pady=1, sticky="w")

# Create the Measurement Status panel
MeasurementStatus = LabelFrame(root, text="Measurement Status", width=410, height=139, padx=10, pady=10)
MeasurementStatus.place(x=246, y=700-375-139)
MeasurementStatus.grid_propagate(False)

# Configurar las columnas para que se expandan proporcionalmente
for col in range(3):  # Total de columnas: 0 a 4
    MeasurementStatus.grid_columnconfigure(col, weight=1)

# Configurar las filas para que se expandan proporcionalmente
for row in range(3):  # Total de filas: 0 a 3
    MeasurementStatus.grid_rowconfigure(row, weight=1)

# Create the Run measurement button
Runmeasurement_btn = Button(MeasurementStatus, text="Run measurement", padx=10, pady=5, command=lambda: threading.Thread(target=Runmeasurement, daemon=True).start())
Runmeasurement_btn.grid(row=0, column=0)

# Create the Stop simulation button
Stopsimulation = Button(MeasurementStatus, text="Stop simulation", padx=10, pady=5, command=Stop)
Stopsimulation.grid(row=1, column=0)

# Create the Progress Bar
progressbar_var = IntVar()
progressbar = ttk.Progressbar(MeasurementStatus, orient='horizontal', mode='determinate', variable=progressbar_var)
progressbar.grid(row=0, column=1)

# Create tghe Progress Bar label
progressbar_label = Label(MeasurementStatus, text='Measurement progress')
progressbar_label.grid(row=1, column=1)

# Crear el panel "Measurement Options"
MeasurementOptions = LabelFrame(root, text="Measurement Options", width=256, height=139, padx=10, pady=10)
MeasurementOptions.place(x=675, y=700-375-139)
MeasurementOptions.grid_propagate(False)  # Evitar que el tamaño cambie

# Configurar las filas y columnas del LabelFrame para centrado
MeasurementOptions.grid_rowconfigure(0, weight=1)  # Fila superior
MeasurementOptions.grid_rowconfigure(1, weight=1)  # Fila del medio
MeasurementOptions.grid_rowconfigure(2, weight=1)  # Fila inferior
MeasurementOptions.grid_columnconfigure(0, weight=1)  # Única columna

# Crear botón para Show Sat Limits
ShowSatLimits = Button(MeasurementOptions, text="Show Sat Limits", command=Showsatlimits)
ShowSatLimits.grid(row=0, column=0, padx=10)

# Crear botón para Curve fitting
Curvefitting = Button(MeasurementOptions, text="Curve fitting", command=Curvefitting)
Curvefitting.grid(row=1, column=0, padx=10)

# Crear botón para FSR
Fsr = Button(MeasurementOptions, text="Full Scale Range", command=Fullscalerange)
Fsr.grid(row=2, column=0, padx=10)

# Crear botón para Str Inv Parameters
StrInv = Button(MeasurementOptions, text="Extra Parameters", command=StrInv)
StrInv.grid(row=3, column=0, padx=10)

# Cargar y redimensionar la imagen de US
US = Image.open("US-marca-principal.png")  # Cambia por la ruta de tu imagen
US_resized = US.resize((100, 90))  # Cambia el tamaño (ancho, alto)
US_photo = ImageTk.PhotoImage(US_resized)
US_label = Label(root, image=US_photo)
US_label.place(x=966, y=700-394-100)  # Cambia la posición (x, y)

# Variable para almacenar el valor seleccionado
scale_mode = StringVar(value="lin")  # Valor inicial

# Crear los radio buttons
lin_button = Radiobutton(root, text="lin", variable=scale_mode, value="lin", command=Linlog)
lin_button.place(x=495, y=700-248-90)  # Empaquetar con un margen vertical

log_button = Radiobutton(root, text="log", variable=scale_mode, value="log", command=Linlog)
log_button.place(x=495, y=700-248-70)

# Creamos Reset button
Reset = Button(root, text="Reset figures", command=Reset)
Reset.place(x=495, y=700-181-22)

# Create label for X Axis
XAxis_label = Label(root, text="X Axis", padx=10, pady=1)
XAxis_label.place(x=495, y=700-150)

# Create the X Axis DropDown
XAxis_options = ['V_DS', 'V_GS']
XAxis_value = StringVar(value=XAxis_options[0])
XAxis = OptionMenu(root, XAxis_value, *XAxis_options)
XAxis.place(x=495, y=700-130)

root.mainloop()
