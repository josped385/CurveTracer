clc;
clear;
close all;

%% Configuración de parámetros de barrido
VPmin = 0;          % Voltaje mínimo para VP
VPmax = 0.8;        % Voltaje máximo para VP
VPsteps = 1000;       % Número de pasos para VP
VNmin = 0;          % Voltaje fijo para VN
VNmax = 0;          % Voltaje fijo para VN
VNsteps = 1000;        % Pasos (fijo en este caso)
Rsense = 1000;      % Resistencia de medición

V_P = [linspace(VPmin, VPmax, VPsteps)'; linspace(VPmax,VPmin,VPsteps)'];
V_N = [linspace(VNmin, VNmax, VNsteps)'; linspace(VNmax,VNmin,VNsteps)'];
V_N_I = linspace(VNmin, VNmax, VNsteps);

% Vector diferencial de voltaje
V_D = V_P - V_N;

%% Configuración del dispositivo DAQ
dq = daq("ni");

% Añadir canales de salida (VP = ao0, VN = ao1)
addoutput(dq, "myDAQ1", 0:1, "Voltage");

% Añadir canal de entrada (V0 = ai1)
addinput(dq, "myDAQ1", "ai1", "Voltage");

%% Escritura y lectura de datos usando readwrite
outputData = [V_P, V_N];  % Configurar las señales de salida
data1 = readwrite(dq, outputData);  % Operación simultánea

% Extraer el voltaje medido V0
V_O_fast = data1.Variables;

% Longitud del vector
n = length(V_O_fast); % Longitud del vector
mid = floor(n/2); % Calculamos el punto medio

V_O_fast1 = V_O_fast(1:mid); % Una curva será hasta la mita
V_D1 = V_D(1:mid); % En el eje X también

V_O_fast2 = flip(V_O_fast(mid+1:end)); % Y aquí la otra mitad, invertida para que vaya de 0 a end
V_D2 = flip(V_D(mid+1:end)); % También en X

V_D = (V_D1 + V_D2)/2;

%% Preprocesamiento: Eliminar duplicados en las curvas
[V_O_fast1_unique, idx1] = unique(V_O_fast1, 'stable'); % Eliminar duplicados en y para la curva 1
V_D1_unique = V_D(idx1); % Ajustar las x correspondientes

[V_O_fast2_unique, idx2] = unique(V_O_fast2, 'stable'); % Eliminar duplicados en y para la curva 2
V_D2_unique = V_D(idx2); % Ajustar las x correspondientes

%% Definir un rango de valores de Y para interpolar
y_min = max(min(V_O_fast1_unique), min(V_O_fast2_unique)); % Máximo mínimo compartido
y_max = min(max(V_O_fast1_unique), max(V_O_fast2_unique)); % Mínimo máximo compartido
n_points = 1000; % Número de puntos para la interpolación
y_values = linspace(y_min, y_max, n_points); % Valores de y para las líneas horizontales

%% Interpolación horizontal
x_interp1 = interp1(V_O_fast1_unique, V_D1_unique, y_values, 'linear', 'extrap'); % x de la curva 1
x_interp2 = interp1(V_O_fast2_unique, V_D2_unique, y_values, 'linear', 'extrap'); % x de la curva 2

%% Calcular el promedio horizontal
x_middle = (x_interp1 + x_interp2) / 2;

%% Graficar VO vs VD
figure;
plot(V_D, V_O_fast1, 'b', 'LineWidth', 1.5); hold on; % Curva 1
hold on
plot(V_D, V_O_fast2, 'r', 'LineWidth', 1.5); % Curva 2
plot(x_middle, y_values, 'g', 'LineWidth', 1.5); % Curva promedio horizontal
xlabel('V_D'); ylabel('V_O');
legend('VPmin -> VPmax', 'VPmax -> VPmin', 'Interpolación', Location='best');
grid on;

%% Graficar ID vs VD (opcional)
figure;
I_D = (V_N_I - y_values) / Rsense;
plot(x_middle, I_D, 'r', 'LineWidth', 1.5);
xlabel('V_D (Voltaje de Diodo) [V]');
ylabel('I_D (Corriente del Diodo) [A]');
title('Curva I_D vs V_D');

%% Liberar recursos
removechannel(dq, [1:length(dq.Channels)]);
disp('Canales eliminados y recursos liberados.');