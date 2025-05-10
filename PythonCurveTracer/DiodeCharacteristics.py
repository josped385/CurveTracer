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

Documentation_window = Help_window = Curvefitting_window = None
V_D = V_O = I_D = VPsteps = VNsteps = IOff_val = R_val = V_N_mean = V_P_mean = None
IVfig = VOVDfig = IVax = IVcanvas = VOVDax = VOVDcanvas = None
sat_line_IVp = sat_line_IVn = sat_line_VOVDp = sat_line_VOVDn = None
cad_line_IV = cad_line_VOVD = None
model_ID = model_VO = V_D_fit = V_D_fit2 = V_O_fit_data = popt_VO = popt_ID = I_D_fit_data = None
IVresiduesfig = VOVDresiduesfig = IVresiduesax = VOVDresiduesax = None
sweep_val = navg = V_N = outer = inner = None
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

def Runmeasurement():

    global V_D, V_O, I_D, VPsteps, VNsteps, IVfig, IVax, IVcanvas, VOVDax, VOVDcanvas, VOVDfig, IOff_val, R_val
    global V_N_mean, V_P_mean, sweep_var, navg, V_N, outer, inner
    # --- Obtener parámetros de los Entry ---
    try:
        VPmin_val = float(VPmin.get())        # Voltaje mínimo para VP
        VPmax_val = float(VPmax.get())        # Voltaje máximo para VP
        VPsteps_val = int(VPsteps.get())      # Número de pasos para VP
        VNmin_val = float(VNmin.get())        # Voltaje fijo para VN
        VNmax_val = float(VNmax.get())        # Voltaje fijo para VN
        VNsteps_val = int(VNsteps.get())      # Pasos (fijo en este caso)
        navg = int(NAveraging.get())          # Número de mediciones para promediar
        IOff_val = float(IOff.get())              # Corriente de offset
    except Exception as e:
        messagebox.showerror("Error", f"Revise the input parameters:\n{e}")
        return
    
    # Parámetros del dispositivo
    device_id = DeviceID.get()
    vp_channel = VP_value.get()
    vn_channel = VN_value.get()
    vo_channel = VO_value.get()
    
    if vp_channel == vn_channel:
            messagebox.showerror('Error', 'The VP and VN channels must be different.')
            return
    elif CheckNImyDAQ_label.cget("bg") == "red":
        messagebox.showerror('Error', 'Turn on the NI device before simulating.')
        return
    elif VPmin_val < -10:
        messagebox.showerror('Error', 'Channels configured for -10 to +10 Volts.')
        return
    elif VPmax_val > 10:
        messagebox.showerror('Error', 'Channels configured for -10 to +10 Volts.')
        return
    elif VNmin_val < -10:
        messagebox.showerror('Error', 'Channels configured for -10 to +10 Volts.')
        return
    elif VNmax_val > 10:
        messagebox.showerror('Error', 'Channels configured for -10 to +10 Volts.')
        return
    elif VNsteps_val < 0:
        messagebox.showerror('Error', 'VNsteps must be a positive value and an integer.')
        return
    elif VPsteps_val < 0:
        messagebox.showerror('Error', 'VPsteps must be a positive value and an integer.')
        return
    
    if sweep_val.get():
    
        # Reiniciamos el evento (aseguramos que no esté marcado)
        stop_event.clear()

        V_P = np.linspace(VPmin_val, VPmax_val, VPsteps_val)
        V_N = np.linspace(VNmin_val, VNmax_val, VNsteps_val)

        # Decidir variable de sweep
        if VPsteps_val > VNsteps_val:
            # Sweep en V_N: para cada VN, barrer todo VP
            sweep_var = 'VN'
            outer = V_N
            inner = V_P
        else:
            # Sweep en V_P: para cada VP, barrer todo VN
            sweep_var = 'VP'
            outer = V_P
            inner = V_N

        # Prealocar matrices de resultados
        V_D = np.zeros((len(outer), len(inner)))
        V_O = np.zeros_like(V_D)
        I_D = np.zeros((len(outer), len(inner)))


        # --- Realizar mediciones y promediar ---
        # Creamos un arreglo para almacenar cada medición
        all_V_O = np.zeros((navg, len(V_D)))
        total_steps = len(outer) * len(inner) * navg
        progressbar["maximum"] = total_steps
        progressbar_var.set(0)
        step = 0
        with Task() as output_task, Task() as input_task:
            # Configurar canales de salida (VP y VN)
            output_task.ao_channels.add_ao_voltage_chan(f"{device_id}/{vp_channel}")
            output_task.ao_channels.add_ao_voltage_chan(f"{device_id}/{vn_channel}")
            # Configurar canal de entrada (V0) en modo diferencial
            input_task.ai_channels.add_ai_voltage_chan(f"{device_id}/{vo_channel}",
                                                    terminal_config=TerminalConfiguration.DIFF)
            
            for i, out_val in enumerate(outer):
                for j, in_val in enumerate(inner):
                    # Definir vp, vn según sweep principal
                    if sweep_var == 'VN':
                        vp = V_P[j]
                        vn = out_val
                    else:
                        vp = out_val
                        vn = V_N[j]

                    # Acumular navg lecturas
                    readings = np.zeros(navg)
                    for k in range(navg):
                        if stop_event.is_set():
                            messagebox.showerror('Simulation stopped', 'The simulation has been stopped')
                            return
                        output_task.write([vp, vn])
                        readings[k] = input_task.read()
                        # -- Actualizar progreso aquí dentro del navg --
                        step += 1
                        progressbar_var.set(step)
                        root.update_idletasks()
                        root.after(1)

                    # Media de las navg lecturas
                    V_O[i, j] = readings.mean()

                    # Calcular V_D = V_P – V_N
                    V_D[i, j] = vp - vn

                    # Promediar las mediciones
                    R_val = float(R.get())
                    I_D[i, j] = (vn - V_O[i, j])/R_val - IOff_val

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
        for i in range(V_D.shape[0]):
            IVax.plot(V_D[i, :], I_D[i, :], linewidth=1,
             label=f"{sweep_var}={outer[i]:.2f} V")
        IVax.set_title("Diode I-V Characteristic", fontsize=10)
        IVax.set_xlabel("V_D [V]", fontsize=10)
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
        for i in range(I_D.shape[0]):
            VOVDax.plot(V_D[i, :], V_O[i, :], linewidth=1,
             label=f"{sweep_var}={outer[i]:.2f} V")
        VOVDax.set_title("Output Voltage", fontsize=10)
        VOVDax.set_xlabel("V_O [V]", fontsize=10)
        VOVDax.set_ylabel("V_D [V]", fontsize=10)
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
        Vsteps_val = np.maximum(VPsteps_val, VNsteps_val)

        # Crear vector de barrido para VP y VN
        V_P = np.linspace(VPmin_val, VPmax_val, Vsteps_val)

        V_N = np.linspace(VNmin_val, VNmax_val, Vsteps_val)

        V_N_mean = np.mean(V_N)
        V_P_mean = np.mean(V_P)

        # Asegurar que V_N tenga la misma forma que V_P
        #if np.isscalar(V_N):
        #    V_N = np.full_like(V_P, V_N)
        # Vector diferencial de voltaje
        V_D = V_P - V_N

        # --- Realizar mediciones y promediar ---
        # Creamos un arreglo para almacenar cada medición
        all_V_O = np.zeros((navg, len(V_D)))
        with Task() as output_task, Task() as input_task:
            # Configurar canales de salida (VP y VN)
            output_task.ao_channels.add_ao_voltage_chan(f"{device_id}/{vp_channel}")
            output_task.ao_channels.add_ao_voltage_chan(f"{device_id}/{vn_channel}")
            # Configurar canal de entrada (V0) en modo diferencial
            input_task.ai_channels.add_ai_voltage_chan(f"{device_id}/{vo_channel}",
                                                    terminal_config=TerminalConfiguration.DIFF)
            
            total_steps = navg * len(V_D)
            progressbar.config(maximum=total_steps)  # O progressbar["maximum"] = total_steps

            # Realizar 'navg' mediciones
            for avg in range(navg):
                for i in range(len(V_D)):
                    if stop_event.is_set():
                        messagebox.showerror('Simulation stopped', 'The simulation has been stopped')
                        return
                    output_task.write([V_P[i], V_N[i]])
                    data = input_task.read()
                    all_V_O[avg, i] = data
                    # Calcular el progreso global:
                    current_step = avg * len(V_D) + i + 1
                    progressbar_var.set(current_step)
                    root.update_idletasks()
                    root.after(1)

        # Promediar las mediciones
        Ioff = np.linspace(IOff_val, IOff_val, len(V_D))
        V_O = np.mean(all_V_O, axis=0)
        R_val = float(R.get())
        I_D = (V_N - V_O)/R_val - Ioff

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
        IVax.plot(V_D, I_D, label="Diode Current", linewidth=1)
        IVax.set_title("Diode I-V Characteristic", fontsize=10)
        IVax.set_xlabel("V_D [V]", fontsize=10)
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
        VOVDax.plot(V_D, V_O, label="Output Voltage", linewidth=1)
        VOVDax.set_title("Output Voltage", fontsize=10)
        VOVDax.set_xlabel("V_O [V]", fontsize=10)
        VOVDax.set_ylabel("V_D [V]", fontsize=10)
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

            global model_ID, model_VO, V_D_fit, V_D_fit2, V_O_fit_data, popt_VO, popt_ID, I_D_fit_data

            if V_D is None or I_D is None or V_O is None:  # Si está definida pero vacía
                messagebox.showerror('Error', 'Run a measurement in order to fit the curve.')
                return

            # --------------------- Ajuste de V_O vs V_D -------------------------
            # Filtrar datos para V_O (ajusta el umbral según corresponda)
            threshold_VO = 0.99 * V_O[-1]
            valid_idx_VO = V_O > threshold_VO
            V_D_fit2 = V_D[valid_idx_VO]      # V_D correspondiente a estos puntos
            V_O_fit_data = V_O[valid_idx_VO]

            # Definir el modelo: f(V_D) = d - e * exp(f * V_D)
            def model_VO(x, d, e, f):
                return d - e * np.exp(f * x)

            # Estimación inicial de parámetros [d, e, f]
            initial_guess_VO = [V_N_mean, R_val * 9.6e-9, 19.5]

            # Ajuste no lineal para V_O vs V_D (se puede ajustar maxfev para mayor precisión)
            popt_VO, pcov_VO = curve_fit(model_VO, V_D_fit2, V_O_fit_data, p0=initial_guess_VO, maxfev=10000)
            d_param, e_param, f_param = popt_VO

            # Calcular el modelo ajustado y el coeficiente de determinación R²
            V_O_model = model_VO(V_D_fit2, *popt_VO)
            SStot2 = np.sum((V_O_fit_data - np.mean(V_O_fit_data))**2)
            SSres2 = np.sum((V_O_fit_data - V_O_model)**2)
            R2_VO = 1 - (SSres2 / SStot2)

            # Estimar errores (desviación estándar de los parámetros) a partir de la matriz de covarianza
            error_d, error_e, error_f = np.sqrt(np.diag(pcov_VO))

            # Asignar resultados a los campos de la interfaz
            d_val.insert(0, f"{d_param: .7g}")
            errord.insert(0, f"{error_d: .7g}")

            e_val.insert(0, f"{e_param: .7g}")
            errore.insert(0, f"{error_e: .7g}")

            f_val.insert(0, f"{f_param: .7g}")
            errorf.insert(0, f"{error_f: .7g}")

            R22.insert(0, f"{R2_VO: .5g}")

            xVOVD_theoretical = np.linspace(np.min(V_D_fit2), np.max(V_D_fit2), 100)
            yVOVD_theoretical = model_VO(xVOVD_theoretical, d_param, e_param, f_param)

            VOVDax.plot(xVOVD_theoretical, yVOVD_theoretical, 'r', label='Vfit')
            VOVDax.legend()
            VOVDcanvas.draw()

            # --------------------- Ajuste de I_D vs V_D -------------------------
            # Filtrar datos para I_D (ajusta el umbral según corresponda)
            #threshold_ID = -threshold_VO/R_val -IOff_val
            threshold_ID = 0.99 * I_D[-1]
            valid_idx_ID = I_D < threshold_ID
            V_D_fit = V_D[valid_idx_ID]      # V_D correspondiente a estos puntos
            I_D_fit_data = I_D[valid_idx_ID]

            # Definir el modelo: f(V_D) = a - b * exp(c * V_D)
            def model_ID(x, a, b, c):
                return a - b * np.exp(c * x)

            # Estimación inicial de parámetros [a, b, c]
            initial_guess_ID = [-9.6e-9, -9.6e-9, 19.5]

            # Ajuste no lineal para V_O vs V_D (se puede ajustar maxfev para mayor precisión)
            popt_ID, pcov_ID = curve_fit(model_ID, V_D_fit, I_D_fit_data, p0=initial_guess_ID, maxfev=10000)
            a_param, b_param, c_param = popt_ID

            # Calcular el modelo ajustado y el coeficiente de determinación R²
            I_D_model = model_ID(V_D_fit, *popt_ID)
            SStot = np.sum((I_D_fit_data - np.mean(I_D_fit_data))**2)
            SSres = np.sum((I_D_fit_data - I_D_model)**2)
            R2_ID = 1 - (SSres / SStot)

            # Estimar errores (desviación estándar de los parámetros) a partir de la matriz de covarianza
            error_a, error_b, error_c = np.sqrt(np.diag(pcov_ID))

            # Asignar resultados a los campos de la interfaz
            a_val.insert(0, f"{a_param: .7g}")
            errora.insert(0, f"{error_a: .7g}")

            b_val.insert(0, f"{b_param: .7g}")
            errorb.insert(0, f"{error_b: .7g}")

            c_val.insert(0, f"{c_param: .7g}")
            errorc.insert(0, f"{error_c: .7g}")

            R2.insert(0, f"{R2_ID: .5g}")

            xIV_theoretical = np.linspace(np.min(V_D_fit), np.max(V_D_fit), 100)
            yIV_theoretical = model_ID(xIV_theoretical, a_param, b_param, c_param)

            IVax.plot(xIV_theoretical, yIV_theoretical, 'r', label='Ifit')
            IVax.legend()
            IVcanvas.draw()

        def Export_fit_data():
            # Crear un diccionario con los nombres y los valores actuales de cada Entry.
            data = {
                "a": a_val.get(),
                "b": b_val.get(),
                "c": c_val.get(),
                "d": d_val.get(),
                "e": e_val.get(),
                "f": f_val.get(),
                "ErrorA": errora.get(),
                "ErrorB": errorb.get(),
                "ErrorC": errorc.get(),
                "ErrorD": errord.get(),
                "ErrorE": errore.get(),
                "ErrorF": errorf.get(),
                "R2": R2.get(),
                "R22": R22.get(),
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
            if (V_D_fit2 is None or V_O_fit_data is None or popt_VO is None or 
                V_D_fit is None or I_D_fit_data is None or popt_ID is None):
                messagebox.showerror("Error", "We can't calculate the residues because the necessary parameters are not defined.")
                return

            # Calcular residuos para V_O vs V_D
            V_O_model = model_VO(V_D_fit2, *popt_VO)
            residues_VO = V_O_fit_data - V_O_model

            # Calcular residuos para I_D vs V_D
            I_D_model = model_ID(V_D_fit, *popt_ID)
            residues_ID = I_D_fit_data - I_D_model

            VOVDresidues = Frame(Curvefitting_window, width=400, height=100, bg="lightgray")
            VOVDresidues.place(x=490, y=220)
            VOVDresidues.grid_propagate(False)
            VOVDresiduesfig = Figure(figsize=(4.5, 2.2), dpi=100)
            VOVDresiduesax = VOVDresiduesfig.add_subplot(111)
            VOVDresiduesax.plot(V_D_fit2, residues_VO, linewidth=1)
            VOVDresiduesax.set_title("Residues for V_O vs V_D", fontsize=10)
            VOVDresiduesax.set_xlabel("V_D [V]", fontsize=10)
            VOVDresiduesax.set_ylabel("Residues for V_O [V]", fontsize=10)
            VOVDresiduesax.grid(True, linestyle='--', linewidth=0.5, color="gray")
            VOVDresiduesfig.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)
            VOVDresiduescanvas = FigureCanvasTkAgg(VOVDresiduesfig, master=VOVDresidues)
            VOVDresiduescanvas.draw()
            VOVDresiduescanvas.get_tk_widget().pack(fill="both", expand=True)

            IVresidues = Frame(Curvefitting_window, width=400, height=100, bg="lightgray")
            IVresidues.place(x=490, y=0)
            IVresidues.grid_propagate(False)
            IVresiduesfig = Figure(figsize=(4.5, 2.2), dpi=100)
            IVresiduesax = IVresiduesfig.add_subplot(111)
            IVresiduesax.plot(V_D_fit, residues_ID, linewidth=1)
            IVresiduesax.set_title("Residues for I_D vs V_D", fontsize=10)
            IVresiduesax.set_xlabel("V_D [V]", fontsize=10)
            IVresiduesax.set_ylabel("Residues for I_D [A]", fontsize=10)
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
        
        def Save_VOVDres():
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
            Curvefitting_window.geometry("1000x500")  # Establecer tamaño de la nueva ventana
            Curvefitting_window.resizable(False, False)

            # Configurar las columnas para que se expandan proporcionalmente
            for col in range(6):  # Total de columnas: 0 a 4
                InputSignalProperties.grid_columnconfigure(col, weight=1)

            # Configurar las filas para que se expandan proporcionalmente
            for row in range(10):  # Total de filas: 0 a 3
                InputSignalProperties.grid_rowconfigure(row, weight=1)
            
            # Crear un menú bar
            menubar = Menu(Curvefitting_window)

            # Crear el menú "File" y sus opciones
            file_menu = Menu(menubar, tearoff=0)
            # Crear el submenú para 'Save Figure'
            save_menu = Menu(menubar, tearoff=0)
            save_menu.add_command(label="Save I-V residue plot", command=Save_IVres)
            save_menu.add_command(label="Save V_O-V_D residue plot", command=Save_VOVDres)
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
            #Export.grid(row=11, columnspan=7)

            #Create Get residues button
            Residues = Button(Curvefitting_window, text="Get residues", command=Get_residues)
            Residues.grid(row=11, columnspan=7)

            # Cargar y redimensionar la ecuación de la intensidad que atraviesa el diodo
            IEquation = Image.open("DiodeCharacteristicsCurrentEquation.png")  # Cambia por la ruta de tu imagen
            #IEquation_resized = IEquation.resize((200, 50))  # Cambia el tamaño (ancho, alto)
            IEquation_photo = ImageTk.PhotoImage(IEquation)
            Curvefitting_window.IEquation_photo = IEquation_photo
            IEquation_label = Label(Curvefitting_window, image=IEquation_photo)
            IEquation_label.grid(row=1, columnspan=7)  # Cambia la posición (x, y)

            # Create label for a
            a_label = Label(Curvefitting_window, text="a (A)", padx=10, pady=10)
            a_label.grid(row=2, column=0, sticky="e")

            # Create entry for a
            a_val = Entry(Curvefitting_window, width=15)
            a_val.grid(row=2, column=1, sticky="w")

            # Create label for errora
            errora_label = Label(Curvefitting_window, text="Error a (A)", padx=10, pady=10)
            errora_label.grid(row=2, column=5, sticky="e")

            # Create entry for errora
            errora = Entry(Curvefitting_window, width=15)
            errora.grid(row=2, column=6, sticky="w")

            # Create label for b
            b_label = Label(Curvefitting_window, text="b (A)", padx=10, pady=10)
            b_label.grid(row=3, column=0, sticky="e")

            # Create entry for b
            b_val = Entry(Curvefitting_window, width=15)
            b_val.grid(row=3, column=1, sticky="w")

            # Create label for errorb
            errorb_label = Label(Curvefitting_window, text="Error b (A)", padx=10, pady=10)
            errorb_label.grid(row=3, column=5, sticky="e")

            # Create entry for errorb
            errorb = Entry(Curvefitting_window, width=15)
            errorb.grid(row=3, column=6, sticky="w")

            # Create label for c
            c_label = Label(Curvefitting_window, text="c (1/V)", padx=10, pady=10)
            c_label.grid(row=4, column=0, sticky="e")

            # Create entry for c
            c_val = Entry(Curvefitting_window, width=15)
            c_val.grid(row=4, column=1, sticky="w")

            # Create label for errorc
            errorc_label = Label(Curvefitting_window, text="Error c (1/V)", padx=10, pady=10)
            errorc_label.grid(row=4, column=5, sticky="e")

            # Create entry for errorc
            errorc = Entry(Curvefitting_window, width=15)
            errorc.grid(row=4, column=6, sticky="w")

            # Create label for R^2
            R2_label = Label(Curvefitting_window, text="R^2", padx=10, pady=10)
            R2_label.grid(row=5, column=3, sticky="e")

            # Create entry for R^2
            R2 = Entry(Curvefitting_window, width=15)
            R2.grid(row=5, column=4, sticky="w")

            # Cargar y redimensionar la ecuación de la intensidad que atraviesa el diodo
            VEquation = Image.open("DiodeCharacteristicsVoltageEquation.png")  # Cambia por la ruta de tu imagen
            #VEquation_resized = VEquation.resize((200, 50))  # Cambia el tamaño (ancho, alto)
            VEquation_photo = ImageTk.PhotoImage(VEquation)
            Curvefitting_window.VEquation_photo = VEquation_photo
            VEquation_label = Label(Curvefitting_window, image=VEquation_photo)
            VEquation_label.grid(row=6, columnspan=7)  # Cambia la posición (x, y)

            # Create label for d
            d_label = Label(Curvefitting_window, text="d (V)", padx=10, pady=10)
            d_label.grid(row=7, column=0, sticky="e")

            # Create entry for d
            d_val = Entry(Curvefitting_window, width=15)
            d_val.grid(row=7, column=1, sticky="w")

            # Create label for errord
            errord_label = Label(Curvefitting_window, text="Error d (V)", padx=10, pady=10)
            errord_label.grid(row=7, column=5, sticky="e")

            # Create entry for errord
            errord = Entry(Curvefitting_window, width=15)
            errord.grid(row=7, column=6, sticky="w")

            # Create label for e
            e_label = Label(Curvefitting_window, text="e (V)", padx=10, pady=10)
            e_label.grid(row=8, column=0, sticky="e")

            # Create entry for e
            e_val = Entry(Curvefitting_window, width=15)
            e_val.grid(row=8, column=1, sticky="w")

            # Create label for errore
            errore_label = Label(Curvefitting_window, text="Error e (V)", padx=10, pady=10)
            errore_label.grid(row=8, column=5, sticky="e")

            # Create entry for errore
            errore = Entry(Curvefitting_window, width=15)
            errore.grid(row=8, column=6, sticky="w")

            # Create label for f
            f_label = Label(Curvefitting_window, text="f (1/V)", padx=10, pady=10)
            f_label.grid(row=9, column=0, sticky="e")

            # Create entry for f
            f_val = Entry(Curvefitting_window, width=15)
            f_val.grid(row=9, column=1, sticky="w")

            # Create label for errorf
            errorf_label = Label(Curvefitting_window, text="Error f (1/V)", padx=10, pady=10)
            errorf_label.grid(row=9, column=5, sticky="e")

            # Create entry for errorf
            errorf = Entry(Curvefitting_window, width=15)
            errorf.grid(row=9, column=6, sticky="w")

            # Create label for R^2
            R22_label = Label(Curvefitting_window, text="R^2", padx=10, pady=10)
            R22_label.grid(row=10, column=3, sticky="e")

            # Create entry for R^2
            R22 = Entry(Curvefitting_window, width=15)
            R22.grid(row=10, column=4, sticky="w")

def Showsatlimits():
    global sat_line_IVp, sat_line_IVn, sat_line_VOVDp, sat_line_VOVDn

    # Verifica que los vectores V_D, I_D y V_O tengan datos
    if V_D is None or I_D is None or V_O is None:
        messagebox.showerror('Error', 'Run a measurement in order to fit the curve.')
        return

    # Verificar que los ejes existan
    if IVax is None or VOVDax is None:
        messagebox.showerror("Error", "Figures are not available.")
        return

    # Si las líneas de saturación ya están representadas, "elimínalas" vaciando sus datos
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
            M, N = V_D.shape
            for i in range(M):
                # El “saturación” depende del valor R_val:
                if R_val < 5000:
                    # calcula vectores de longitud N
                    isatp = 2e-3
                    isatn = -isatp
                    vsatp = outer[i] + 2e-3 * R_val
                    vsatn = outer[i] - 2e-3 * R_val
                else:
                    # valores constantes si R_val grande
                    vsatp = 10
                    vsatn = -10
                    isatp = 10 / R_val
                    isatn = isatp

                # crear vectores constantes
                Vsatp_line = np.full(N, vsatp)
                Vsatn_line = np.full(N, vsatn)
                Isatp_line = np.full(N, isatp)
                Isatn_line = np.full(N, isatn)

                #sat_line_IVp   = []
                #sat_line_IVn   = []
                #sat_line_VOVDp = []
                #sat_line_VOVDn = []

                # plotear sobre cada fila de V_D[i, :]
                sat_line_VOVDp = VOVDax.plot(V_D[i, :], Vsatp_line, 'k--', label='Vsat+')
                #sat_line_VOVDp.extend(lines)
                sat_line_VOVDn = VOVDax.plot(V_D[i, :], Vsatn_line, 'k--', label='Vsat-')
                #sat_line_VOVDn.extend(lines)
                sat_line_IVp = IVax.plot(V_D[i, :], Isatp_line, 'k--', label='Isat+')
                #sat_line_IVp.extend(lines)
                sat_line_IVn = IVax.plot(V_D[i, :], Isatn_line, 'k--', label='Isat-')
                #sat_line_IVn.extend(lines)
        else:
            # Crear los vectores para las líneas de saturación:
            N = len(V_D)  # Se usan N puntos para la línea (realmente bastan 2, pero se usa N para compatibilidad)
            if R_val < 5000:
                isat = 2e-3
                vsat = 2e-3 * R_val
            else:
                vsat = 10
                isat = vsat/R_val
            Vsatp = np.linspace(vsat, vsat, N)   # Línea horizontal en vsat
            Vsatn = np.linspace(-vsat, -vsat, N) # Línea horizontal en -vsat
            Isatp = np.linspace(isat, isat, N)   # Línea horizontal en Isat
            Isatn = np.linspace(-isat, -isat, N)  # Línea horizontal en -Isat

            # Añadir las líneas de saturación a la gráfica
            sat_line_IVp = IVax.plot(V_D, Isatp, 'r--', label="Isat+")
            sat_line_IVn = IVax.plot(V_D, Isatn, 'r--', label="Isat-")
            sat_line_VOVDp = VOVDax.plot(V_D, Vsatp, 'r--', label="Vsat+")
            sat_line_VOVDn = VOVDax.plot(V_D, Vsatn, 'r--', label="Vsat-")
    
    # Actualizar las leyendas
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
        # Crear un nuevo eje
        IVax = IVfig.add_subplot(111)
        IVax.set_title("Diode I-V Characteristic", fontsize=10)
        IVax.set_xlabel("V_D [V]", fontsize=10)
        IVax.set_ylabel("I_D [A]", fontsize=10)
        IVax.grid(True, linestyle='--', linewidth=0.5, color="gray")
        IVcanvas.draw()  # Redibuja el canvas
        # Limpiar las variables de datos y líneas
        I_D = None  # Limpiamos I_D
        V_D = None  # Limpiamos V_D
        sat_line_IVp = None
        sat_line_IVn = None
    except Exception as e:
        print("The I_D-V_D figure couldn't be cleaned:", e)

    try:
        # Limpiar la figura VOVD (Output Voltage)
        VOVDfig.clf()  # Borra todos los ejes y trazos
        # Crear un nuevo eje
        VOVDax = VOVDfig.add_subplot(111)
        VOVDax.set_title("Output Voltage", fontsize=10)
        VOVDax.set_xlabel("V_O [V]", fontsize=10)
        VOVDax.set_ylabel("V_D [V]", fontsize=10)
        VOVDax.grid(True, linestyle='--', linewidth=0.5, color="gray")
        VOVDcanvas.draw()  # Redibuja el canvas
        V_O = None  # Limpiamos V_O
        sat_line_VOVDp = None
        sat_line_VOVDn = None
    except Exception as e:
        print("The V_O-V_D figure couldn't be cleaned:", e)

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
        "VPmin": VPmin.get(),
        "VPmax": VPmax.get(),
        "VPsteps": VPsteps.get(),
        "VNmin": VNmin.get(),
        "VNmax": VNmax.get(),
        "VNsteps": VNsteps.get(),
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

def Export_VOVD_Data():
    # Se asume que IVax es un objeto Axes de Matplotlib (por ejemplo, global)
    lines = VOVDax.get_lines()
    if not lines:
        messagebox.showerror("Error", "Couldn't find any data in Diode V_O-V_D Characteristic figure.")
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
            messagebox.showerror('Error', "In order to load data, run a measurement to create a figure. Right now, the axes don't exist, so the program doesn't know where to represent the data.")
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

root = Tk()
root.title("Diode Characteristics Application")
root.iconbitmap('US-marca-principal.ico')
root.geometry("1100x700")
root.resizable(False, False)

# Crear un menú bar
menubar = Menu(root)

# Crear el menú "File" y sus opciones
file_menu = Menu(menubar, tearoff=0)
# Crear el submenú para 'Save Figure'
save_menu = Menu(menubar, tearoff=0)
save_menu.add_command(label="Save I-V Diode Characteristic", command=Save_IV)
save_menu.add_command(label="Save V_O-V_D Diode Characteristic", command=Save_VOVD)
file_menu.add_cascade(label="Save Figure...", menu=save_menu)
file_menu.add_command(label="Save Config", command=Save_config)
rename_menu = Menu(menubar, tearoff=0)
rename_menu.add_command(label="Rename in I-V Diode Characteristic", command=Rename_IV)
rename_menu.add_command(label="Rename in V_O-V_D Diode Characteristic", command=Rename_VOVD)
file_menu.add_cascade(label="Rename Curve...", menu=rename_menu)
export_menu = Menu(menubar, tearoff=0)
export_menu.add_command(label="Export I-V Diode Characteristic Data", command=Export_IV_Data)
export_menu.add_command(label="Export V_O-V_D Diode Characteristic Data", command=Export_VOVD_Data)
file_menu.add_cascade(label="Export Data...", menu=export_menu)
hide_menu = Menu(menubar, tearoff=0)
hide_menu.add_command(label="Hide I-V Diode Characteristic Curve", command=Hide_IV)
hide_menu.add_command(label="Hide V_O-V_D Diode Characteristic Curve", command=Hide_VOVD)
file_menu.add_cascade(label="Hide Curve...", menu=hide_menu)
show_menu = Menu(menubar, tearoff=0)
show_menu.add_command(label="Show I-V Diode Characteristic Curve", command=Show_IV)
show_menu.add_command(label="Show V_O-V_D Diode Characteristic Curve", command=Show_VOVD)
file_menu.add_cascade(label="Show Curve...", menu=show_menu)
load_menu = Menu(menubar, tearoff=0)
load_menu.add_command(label="Load I-V Diode Characteristic Curve", command=Load_IV)
load_menu.add_command(label="Load V_O-V_D Diode Characteristic Curve", command=Load_VOVD)
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

# Create label for VPmin
VPmin_label = Label(InputSignalProperties, text="VPmin (V)", padx=10, pady=5)
VPmin_label.grid(row=0, column=0, sticky="e")  # Alinear a la derecha

# Create the VPmin Edit Field
VPmindefault = IntVar()
VPmindefault.set(0)
VPmin = Entry(InputSignalProperties, width=15, textvariable=VPmindefault)
VPmin.grid(row=0, column=1, sticky="w")  # Alinear a la izquierda

# Create label for VPmax
VPmax_label = Label(InputSignalProperties, text="VPmax (V)", padx=10, pady=5)
VPmax_label.grid(row=2, column=0, sticky="e")  # Alinear a la derecha

# Create the VPmax Edit Field
VPmaxdefault = IntVar()
VPmaxdefault.set(0.8)
VPmax = Entry(InputSignalProperties, width=15, textvariable=VPmaxdefault)
VPmax.grid(row=2, column=1, sticky="w")  # Alinear a la izquierda

# Create label for VPsteps
VPsteps_label = Label(InputSignalProperties, text="VPsteps", padx=10, pady=5)
VPsteps_label.grid(row=3, column=0, sticky="e")  # Alinear a la derecha

# Create the VPsteps Edit Field
VPstepsdefault = IntVar()
VPstepsdefault.set(25)
VPsteps = Entry(InputSignalProperties, width=15, textvariable=VPstepsdefault)
VPsteps.grid(row=3, column=1, sticky="w")  # Alinear a la izquierda

# Create label for VNmin
VNmin_label = Label(InputSignalProperties, text="VNmin (V)", padx=10, pady=5)
VNmin_label.grid(row=0, column=3, sticky="e")  # Alinear a la derecha

# Create the VNmin Edit Field
VNmindefault = IntVar()
VNmindefault.set(0)
VNmin = Entry(InputSignalProperties, width=15, textvariable=VNmindefault)
VNmin.grid(row=0, column=4, sticky="w")  # Alinear a la izquierda

# Create label for VNmax
VNmax_label = Label(InputSignalProperties, text="VNmax (V)", padx=10, pady=5)
VNmax_label.grid(row=2, column=3, sticky="e")  # Alinear a la derecha

# Create the VNmax Edit Field
VNmaxdefault = IntVar()
VNmaxdefault.set(0)
VNmax = Entry(InputSignalProperties, width=15, textvariable=VNmaxdefault)
VNmax.grid(row=2, column=4, sticky="w")  # Alinear a la izquierda

# Create label for VNsteps
VNsteps_label = Label(InputSignalProperties, text="VNsteps", padx=10, pady=5)
VNsteps_label.grid(row=3, column=3, sticky="e")  # Alinear a la derecha

# Create the VNsteps Edit Field
VNstepsdefault = IntVar()
VNstepsdefault.set(25)
VNsteps = Entry(InputSignalProperties, width=15, textvariable=VNstepsdefault)
VNsteps.grid(row=3, column=4, sticky="w")  # Alinear a la izquierda

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

# Create label for VP
VP_label = Label(NImyDAQPorts, text="VP", padx=10, pady=1)
VP_label.grid(row=0, column=0, sticky="e")

# Create the VP DropDown
VP_options = ['ao0', 'ao1']
VP_value = StringVar(value=VP_options[0])
#VP_value.set(VP_options[0])
VP = OptionMenu(NImyDAQPorts, VP_value, *VP_options)
VP.grid(row=0, column=1, padx=10, pady=1, sticky="w")

# Create label for VN
VN_label = Label(NImyDAQPorts, text="VN", padx=10, pady=1)
VN_label.grid(row=1, column=0, sticky="e")

# Create the VN DropDown
VN_options = ['ao0', 'ao1']
VN_value = StringVar(value=VN_options[1])
#VN_value.set(VN_options[1])
VN = OptionMenu(NImyDAQPorts, VN_value, *VN_options)
VN.grid(row=1, column=1, padx=10, pady=1, sticky="w")

# Create label for VO
VO_label = Label(NImyDAQPorts, text="VO", padx=10, pady=1)
VO_label.grid(row=3, column=0, sticky="e")

# Create the VN DropDown
VO_options = ['ai0', 'ai1']
VO_value = StringVar(value=VO_options[1])
#VO_value.set(VO_options[1])
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

## Creamos checkbox para Hold on
#holdon_var = IntVar(value = 0)
#Holdon = Checkbutton(root, text="Hold on", variable=holdon_var)
#Holdon.place(x=495, y=700-227-22)

# Creamos Reset button
Reset = Button(root, text="Reset figures", command=Reset)
Reset.place(x=495, y=700-181-22)

root.mainloop()