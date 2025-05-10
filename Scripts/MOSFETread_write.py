import numpy as np
import matplotlib.pyplot as plt
from nidaqmx import Task
from nidaqmx.constants import TerminalConfiguration

# Configuración de parámetros de barrido
VDmin = 2.7           # Voltaje mínimo para VD
VDmax = 2.7         # Voltaje máximo para VD
Vsteps = 25        # Número de pasos para VD
VGmin = 0           # Voltaje fijo para VG
VGmax = 5           # Voltaje fijo para VG

V_D = np.linspace(VDmin, VDmax, Vsteps)  # Barrido para VD
V_G = np.linspace(VGmin, VGmax, Vsteps)  # VG constante en este caso

# Ajustar tamaños de VD y VG para combinar
if np.isscalar(V_G):
    V_G = np.full_like(V_D, V_G)

# Vector diferencial de voltaje
V_Diff = V_D - V_G

# Inicializar el voltaje de salida
V_O = np.zeros_like(V_Diff)

# Configuración del dispositivo DAQ
with Task() as output_task, Task() as input_task:
    # Añadir canales de salida (VD = ao0, VG = ao1)
    output_task.ao_channels.add_ao_voltage_chan("myDAQ1/ao0")
    output_task.ao_channels.add_ao_voltage_chan("myDAQ1/ao1")

    # Añadir canal de entrada (V0 = ai1) con configuración diferencial
    input_task.ai_channels.add_ai_voltage_chan("myDAQ1/ai1", terminal_config=TerminalConfiguration.DIFF)

    # Escritura y lectura de datos
    for i in range(len(V_Diff)):
        # Escribir valores de VD y VG
        output_task.write([V_D[i], V_G[i]])

        # Leer el voltaje de salida (V0)
        data = input_task.read()
        V_O[i] = data

# Graficar VO vs VD
plt.figure()
if VGmin == VGmax:
    plt.plot(V_D, V_O, 'b', linewidth=1.5)
    plt.xlabel('V_D [V]')
else:
    plt.plot(V_G, V_O, 'b', linewidth=1.5)
    plt.xlabel('V_G [V]')
plt.ylabel('V_O [V]')
plt.title('Voltaje de salida')
plt.grid(True)

# Graficar ID vs VD
plt.figure()
I_D = (V_O - V_D)/1000
if VGmin == VGmax:
    plt.plot(V_D, I_D, 'b', linewidth=1.5)
    plt.xlabel('V_D [V]')
else:
    plt.plot(V_G, I_D, 'b', linewidth=1.5)
    plt.xlabel('V_G [V]')
plt.ylabel('I_D [A]')
plt.title('Curva característica I-V del MOSFET')
plt.grid(True)

# Mostrar las gráficas
plt.show()