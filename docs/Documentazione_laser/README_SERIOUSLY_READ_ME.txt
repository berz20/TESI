Developed 2016 by Nikolas Tomek (tomek.qudi@gmail.com), Institute for Quantum Optics, Ulm University

PLEASE READ THIS WHOLE FILE BEFORE USING THE MODULE!!!

###############################
General stuff about the driver:
###############################

This board is a very early and crude realization of a fast switching laser diode driver
that can be controlled by a TTL (tested) or LVDS (not tested) input. 
The design is built around the "iC-HG iCSY HG8M" driver module from "iC-Haus".
The idea is that you can switch a laser diode on and off very quickly (~couple ns).
The desired power-on level can be adjusted using the potentiometer POT12.
To enhance the rise time at the expense of extinction ratio you can enable a second
current channel that is biasing the laser diode with a constant current. 
This current can be adjusted using the potentiometer POT36.

The driver was tested in the Institute for Quantum Optics using a "Roithner LD-520-120MG" laser diode.
This diode was thermally connected to an aluminum block that can be temperature controlled. 
As TTL driver the output of a Spartan-6 FPGA was used with LVCMOS 3.3V standard.

To understand the working principle of the driver it is highly recommended to read and understand 
the datasheet for the iC-Haus chip that was used as heart of the driver. The "iC-HG".

The performance and stability of the diode is highly influenced by the following section.


######################
COOLING --- IMPORTANT:
######################

Before you start with anything please note that laser diodes are highly vulnerable to thermal damage.
The whole characteristics of the laser diode like center wavelength and internal resistance are 
extremely temperature dependent.
So first of all you want to make sure that the Laser diode does not exceed ~60°C during operation. 
To do so, ensure that the diode has good thermal contact to a suitable heatsink.
If you are interested in stable power levels you even need to adapt temperature control and stabilization.

The SOIC-8 package of the LDO supplying the anode voltage for the laser diode also must be connected 
to a heatsink. 


#######################
Pin header descriptions:
#######################

----
JP1:
----
This 3-pol pin header is for monitoring the following signals:

pin 1 --- open drain error monitor output of the iC-HG
pin 2 --- current control voltage of POT12. This voltage controls the "on"-current.
pin 3 --- current control voltage of POT36. This voltage controls the bias current.

----
JP2:
----
This 8-pol pin header is for configuring the enable (ENx) inputs of the "iC-HG iCSY HG8M" via the ELVDS input selector. 
Use a single jumper to chose ONLY ONE of the marked configs (1-4):

jumper pos. 1 --- fast single-ended TTL mode. connects ELVDS to GND.
jumper pos. 2 --- slow single-ended TTL mode. connects ELVDS to 30% VDD.
jumper pos. 3 --- slow differential LVDS mode. connects ELVDS to 70% VDD.
jumper pos. 4 --- fast differential LVDS mode. connects ELVDS to VDD.

----
JP3:
----
This 4-pol pin header is for connecting the SMA inputs to ground when they are not actively driven high or low.
Connect for example the EN_N to GND if you are using TTL mode instead of LVDS.
Or connect both to GND if you want to adjust the bias current through the laser diode (with POT36).

----
JP4:
----
This 3-pol pin header is for connecting the EN signal for the second channel to either GND or VDD.
By doing so you effectively enable or disable the second channel which can be used to bias the laser diode
with a constant current. This current is adjustable with POT36. 
Exactly one config must be chosen by setting a jumper.

connect pins 1 and 2 --- channel 2 enabled. EN connected to VDD.
connect pins 2 and 3 --- channel 2 disabled. EN connected to GND.

----
FAN:
----
This 3-pol pin header can be used to power for example a PC fan. 
It is connected to the laser diode anode LDO which can produce enough "extra juice" to power a small device like a PC fan.

pin 1 --- not connected
pin 2 --- adjustable voltage output of the MIC39102
pin 3 --- GND


#################################
How to set things up (typically):
#################################

0)	DO NOT CONNECT THE LASER DIODE YET.
1)	Put a jumper on the two pins marked as "1" on JP2. This enables fast TTL mode.
2)	connect both SMA connectors to ground by using JP3.
3)	Use a jumper to connect pins 1 and 2 of JP4 to enable the bias channel.
4)	Connect a 12V-DC (min. 1A) power supply to the DC jack of the board.
5)	Monitor the voltage of LDA to GND (printed on PCB). Adjust the potentiometer R9 of the carrier board
	until you have a voltage that is suitable as anode voltage for your laser diode.
	(~ 6V for Roithner LD-520-120MG)
6)	Check voltages of pins 2 and 3 to GND of JP1 
	to make sure that POT12 and POT36 on the iC-Haus board are turned down to (almost) zero.
7)	Disconnect the power supply.
8)	Connect the laser diode.
9)	Power up the board.
10)	Carefully turn up POT36 on the iC-Haus board until you reach your desired bias current for the laser diode.
11) Remove the jumper on the SMA connector labled "EN1_P" and draw the input to constant TTL high (5V)
12) Carefully turn up POT12 on the iC-Haus board until you reach your desired "on"-current for the laser diode.
13) Send TTL pulses to the SMA connection "EN1_P" and check if everything works as intended.

NOTE:	The threshold current of a laser diode shifts as soon as the temperature of the diode changes.
		Meaning the adjusted current levels will lead to different currents (and optical power) as soon as 
		the diode is heating up during operation.
		Sometimes you may also want to adjust the anode voltage using R9 if you can not reach a certain power level.
		BUT BE CAREFUL!		