clc;
clear;
close all;

%% Configuración de parámetros de barrido
VDmin = 0;          % Voltaje mínimo para VP
VDmax = 5;        % Voltaje máximo para VP
VDsteps = 25;       % Número de pasos para VP
VGmin = 2.7;          % Voltaje fijo para VN
VGmax = 2.7;          % Voltaje fijo para VN
VGsteps = 25;        % Pasos (siempre fijo en este caso)
R = 1000;           % Valor de la resistencia de sensado

V_D = linspace(VDmin, VDmax, VDsteps)';  % Barrido para VP
V_G = linspace(VGmin, VGmax, VGsteps)';  % VN constante en este caso

% Ajustar tamaños de VP y VN para combinar
if isscalar(V_G)
    V_G = V_G * ones(size(V_D));
end

% Vector diferencial de voltaje
V_Diff = V_D - V_G;

%% Configuración del dispositivo DAQ
dq = daq("ni");

% Añadir canales de salida (VD = ao0, VG = ao1)
addoutput(dq, "myDAQ1", 0:1, "Voltage");

% Añadir canal de entrada (V0 = ai1)
addinput(dq, "myDAQ1", "ai1", "Voltage");

%% Escritura y lectura de datos
V_O = zeros(size(V_Diff));  % Inicializar el voltaje de salida

for i = 1:length(V_Diff)
    % Escribir valores de VD y VG
    write(dq, [V_D(i), V_G(i)]);
    
    % Leer el voltaje de salida (V0)
    data = read(dq, 1);  % Leer 1 muestra
    V_O(i) = data.Variables;  % Almacenar el voltaje de salida
end

%% Graficar VO vs VD o VG
figure;
if VDmin == VDmax
    plot(V_G, V_O, 'b', 'LineWidth', 1.5);
    xlabel('V_G [V]');
else
    plot(V_D, V_O, 'b', 'LineWidth', 1.5);
    xlabel('V_D [V]');
end
ylabel('V_O [V]');
title('Tensión de salida del MOSFET');
grid on;

%% Graficar I_D vs VD o VG
Id = (V_O - V_D)/R;
figure;
if VDmin == VDmax
    plot(V_G, Id, 'b', 'LineWidth', 1.5);
    xlabel('V_G [V]');
else
    plot(V_D, Id, 'b', 'LineWidth', 1.5);
    xlabel('V_D [V]');
end
ylabel('I_D [V]');
title('Curva característica I-V del MOSFET');
grid on;

%% Eliminar canales y liberar recursos
removechannel(dq, 1:length(dq.Channels));
disp('Canales eliminados y recursos liberados.');