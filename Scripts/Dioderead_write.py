import numpy as np
import matplotlib.pyplot as plt
from nidaqmx import Task
from nidaqmx.constants import TerminalConfiguration

# Configuración de parámetros de barrido
VPmin = 0           # Voltaje mínimo para VP
VPmax = 0.8         # Voltaje máximo para VP
VPsteps = 1000        # Número de pasos para VP
VNmin = 0           # Voltaje fijo para VN
VNmax = 0           # Voltaje fijo para VN
VNsteps = 1000         # Pasos (siempre fijo en este caso)

V_P = np.linspace(VPmin, VPmax, VPsteps)  # Barrido para VP
V_N = np.linspace(VNmin, VNmax, VNsteps)  # VN constante en este caso

# Ajustar tamaños de VP y VN para combinar
if np.isscalar(V_N):
    V_N = np.full_like(V_P, V_N)

# Vector diferencial de voltaje
V_D = V_P - V_N

# Inicializar el voltaje de salida
V_O = np.zeros_like(V_D)

# Configuración del dispositivo DAQ
with Task() as output_task, Task() as input_task:
    # Añadir canales de salida (VP = ao0, VN = ao1)
    output_task.ao_channels.add_ao_voltage_chan("myDAQ1/ao0")
    output_task.ao_channels.add_ao_voltage_chan("myDAQ1/ao1")

    # Añadir canal de entrada (V0 = ai1) con configuración diferencial
    input_task.ai_channels.add_ai_voltage_chan("myDAQ1/ai1", terminal_config=TerminalConfiguration.DIFF)

    # Escritura y lectura de datos
    for i in range(len(V_D)):
        # Escribir valores de VP y VN
        output_task.write([V_P[i], V_N[i]])

        # Leer el voltaje de salida (V0)
        data = input_task.read()
        V_O[i] = data

# Graficar VO vs VD
plt.figure()
plt.plot(V_D, V_O, 'b', linewidth=1.5)
plt.xlabel('V_D (Voltaje de Diodo) [V]')
plt.ylabel('V_O (Voltaje de Salida) [V]')
plt.title('Curva V_O vs V_D')
plt.grid(True)

# Graficar ID vs VD
plt.figure()
I_D = (V_N-V_O) / 1000
plt.plot(V_D, I_D, 'b', linewidth=1.5)
plt.xlabel('V_D (Voltaje de Diodo) [V]')
plt.ylabel('I_D (Corriente del diodo) [A]')
plt.title('Curva I_D vs V_D')
plt.grid(True)

# Mostrar las gráficas
plt.show()