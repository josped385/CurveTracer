# CurveTracer
This repository contains applications and scripts that helps the user to plot the characteristic curves of diodes and MOSFET transistors using NI myDAQ. Some additional features are built into the applications; curve fitting, extraction of parameters, saturation limits, etc.

The application has been programmed in two languages: MATLAB and Python.

How to navigate this repository:

- MATLABCurveTracer: this folder contains all the code for the Graphic User Interface (GUI) application designed with App Designer in MATLAB, as well as the different images and data required for the GUI.
- PythonCurveTracer: this folder contains all the code for the GUI application designed with tkinter in Python, as well as the different images and data required for the GUI.
- Scripts: this folder contains the code used to obtain the curves of the diode and the MOSFET without a GUI. It is much simpler and direct, it is made just to give the user the curves.

Circuits used:

For the diode, I used a LM741, a 1N4007 diode, cables and resistors (1kOhms and 30kOhms, mainly). The circuit used is the following:

![imagen](https://github.com/user-attachments/assets/05830d7f-5e03-4768-b4b3-468e1fbd3f48)

For the MOSFET, I used a LM741, a MC4007UB MOSFET, cables and resistors (1kOhms and 30kOhms, mainly). The circuit used is the following:

![imagen](https://github.com/user-attachments/assets/7bf3a4fd-c645-47cd-b1c3-40c20e498170)

# MATLAB Curve Tracer

The first file the user can access is the Curve Tracer App, which provides access to the Help App of Curve Tracer, the Documentation App, and both the Diode and MOSFET Characteristics applications. If the user accesses the Diode Characteristics application, this user interface will be shown:

![imagen](https://github.com/user-attachments/assets/2a0d9119-8dd8-41b4-8511-57cdb3be74e0)

There is an additional window that can be found when pressing the Curve fitting button, which is:

![imagen](https://github.com/user-attachments/assets/777d9c4d-6d87-4a60-9f84-b662aaf63c40)

This are the functionalities of the app:

NI myDAQ: In this panel the user can make sure that the myDAQ is connected properly and MATLAB is detecting the hardware. In the case that the device is not connected or not connected properly, an error message will be displayed. The user can also introduce the ID name of the device, so it can be detected properly.

NI myDAQ Ports: In this panel the user can control the channels employed in myDAQ as inputs and outputs to define what ports are used as VP, VN and VO.

Input Signal Properties: In this panel the user can define the characteristics of the signal that will be sent to the circuit connected to the myDAQ, such as:
- VPmin: minimum value of the VP signal (see Schematic in the help toolbar).
- VPmax: maximum value of the VP signal.
- VPsteps: number of divisions in which the vector is going to define the signal sent to the circuit.
Analogously:
- VNmin: minimum value of the VN signal.
- VNmax: maximum value of the VN signal.
- VNsteps: number of divisions in which the vector is going to define the signal sent to the circuit.
- Sweep: this checkbox allows the user to plot multiple graphs by creating a sweep. if VPsteps > VNsteps, it creates VNsteps curves with constant VN values. The same effect happens with VPsteps.
NOTE: the program takes the highest value between VPsteps and VNsteps and make the vectors VP and VN with that value, so the vectors are made with compatible sizes for the operations.

Measurement Status: In this panel the user can control the simulation proceeding. Functionalities available in this panel:
- Run measurement: with this button the user can start the simulation.
- Stop simulation: with this buttton the user can stop the simulation at any moment.
- Measurement Status (%): in this gauge, the user can see the percentage that represents how much is left to complete the simulation (only available for the Slow Mode (the Fast Mode is not selected)).

Measurement Properties: In this panel the user can modify some parameters relevant to the simulation and circuit:
- R: in this field the user can modify the value of the resistor employed in the circuit.
- NAveraging: in this field the user can define the number of simulations that can be done before averaging and representing the averaged result.
- IOff: it substract the quantity IOff to the current plotted in the Diode I-V Characteristics graph to correct offset current.

Measurement Options: In this panel the user can modify some aspects relevant to the simulation that is going to be done.
- Show Sat Limits: when pressed, this button shows in both graphs the saturation limits of current and voltage of the circuit. The user can press this button again, when it says 'Hide Sat Limits', to hide the saturation limits.
- Curve fitting: this button gives access to the user to a panel in which the user can model the last curve represented as an exponential function, and the parameters and the errors are given. It also displays the correlation factor of the adjustment, so the user can have an idea of how good the curve fitting is. In the graphs, a function with the parameters calculated is represented, with names Ifit and Vfit, so the user can have a visual idea of the calculation realised by the curve fitting button.
    - In this panel the user can fit the data, obtain the residues, export the fitting parameters in .csv format and save the residues plots.
- Full Scale Range: it shows the Full Scale Range (FSR) of the measurement.

If you click the menu above the GUI, you can have access to some options regarding the plots and the data:

- Rename: the user can rename any curve represented, so the graph can be better studied and understood by the user.
- Save as: the user can use this option to same the image obtained in the graph, in various formats.
- Export Data to Workspace: the user can export the data of the curves to the Workspace of MATLAB, so further analysis can be performed. The data is exported with the name of the curve, in a table with two columns, the first one are the x-values, and the second are the y-values.
- Load data: with this option, the user can load data into the graph, so comparisons and further analysis can be made.
- Hide Graph: with this option, the user can hide the graph selected.
- Show Graph: if the user hid a graph with the previous options, the user can show again that graph with this one.
- Save config: with this option, the user can save in .csv format the VPmin, VPmax, VNmin, R, IOff, etc. data.

If the user accesses the MOSFET Characteristics application, this user interface will be shown:

![imagen](https://github.com/user-attachments/assets/a4c72ab1-1cfe-4e6f-96c1-9b95fce57f00)

There are additional windows, just like the case of the diode:

Curve fitting:

![imagen](https://github.com/user-attachments/assets/4a2c3b29-cf55-404a-ac4b-ea358ad427d9)

Extra Parameters: degradation of mobility (when V_DS is in the X Axis):

![imagen](https://github.com/user-attachments/assets/d34533a3-9606-4653-80c3-dd3190ce54b7)

Extra Parameters: strong inversion parameters (when V_GS is in the X Axis):

![imagen](https://github.com/user-attachments/assets/324b7c7f-3b2d-4d10-9bfa-3b84bb78d96e)

The functionalities of this app is similar to that of the diode; therefore, we only add the functionalities of the Extra Parameters window:

- Extra Parameters:In this panel the user can obtain extra parameters for ohmic and saturation regions, as well as strong inversion regions using level 3 model of the transistor. The parameters that can be obtained for the ohmic and saturation regions are UO, THETA and VTO, defined in another panel of the Help App. The errors of the fit and the correlation coefficients of the adjustment are also displayed. For the strong inversion parameters, we don't have errors or correlation coefficients. The coefficients are GAMMA, KAPPA, PHI and THETA2 (also shown in this Help App). The substrate effect can be specified to obtain the corresponding coefficients.

# Python Curve Tracer

This application works exactly like the MATLAB application, only with a few differences:
- There is no "Hold on" Check box, so each time a measurement is made the curves shown in the figures are deleted.
- Nothing will happen if the user right-clicks the figures; in order to make changes to the axis, the user needs to access the menu toolbar on top of the application.
- There is no Fast Mode in ANY app. It is not available in the Diode Characteristics app, and it is not available in the MOSFET Characteristics app.
- The strong inversion parameters window works specially bad; the results displayed are not accurate, this is a functionality that needs to be fixed.
