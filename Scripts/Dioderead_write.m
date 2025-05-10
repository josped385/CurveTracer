clc;
clear;
close all;

%% Configuración de parámetros de barrido
VPmin = 0;          % Voltaje mínimo para VP
VPmax = 0.8;        % Voltaje máximo para VP
VPsteps = 100;       % Número de pasos para VP
VNmin = 0;          % Voltaje fijo para VN
VNmax = 0;          % Voltaje fijo para VN
VNsteps = 100;        % Pasos (siempre fijo en este caso)

V_P = linspace(VPmin, VPmax, VPsteps)';  % Barrido para VP
V_N = linspace(VNmin, VNmax, VNsteps)';  % VN constante en este caso

% Ajustar tamaños de VP y VN para combinar
if isscalar(V_N)
    V_N = V_N * ones(size(V_P));
end

% Vector diferencial de voltaje
V_D = V_P - V_N;

%% Configuración del dispositivo DAQ
dq = daq("ni");

% Añadir canales de salida (VP = ao0, VN = ao1)
addoutput(dq, "myDAQ1", 0:1, "Voltage");

% Añadir canal de entrada (V0 = ai1)
addinput(dq, "myDAQ1", "ai1", "Voltage");

%% Escritura y lectura de datos
V_O = zeros(size(V_D));  % Inicializar el voltaje de salida

for i = 1:length(V_D)
    % Escribir valores de VP y VN
    write(dq, [V_P(i), V_N(i)]);
    
    % Leer el voltaje de salida (V0)
    data = read(dq, 1);  % Leer 1 muestra
    V_O(i) = data.Variables;  % Almacenar el voltaje de salida
end

%% Graficar VO vs VD
figure;
plot(V_D, V_O, 'b', 'LineWidth', 1.5);
xlabel('V_D (Voltaje de Diodo) [V]');
ylabel('V_O (Voltaje de Salida) [V]');
title('Curva V_O vs V_D');
grid on;

%% Graficar ID vs VD

figure;
I_D = (V_N-V_O)/1000;
plot(V_D, I_D, 'b', 'LineWidth', 1.5)
xlabel('V_D (Voltaje de Diodo) [V]');
ylabel('I_D (Corriente del diodo) [A]');
title('Curva I_D vs V_D');
grid on;

%% Eliminar canales y liberar recursos
removechannel(dq, [1:length(dq.Channels)]);
disp('Canales eliminados y recursos liberados.');



