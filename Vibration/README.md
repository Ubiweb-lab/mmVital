# mmVital
Simulation of Vibration to compensate misalignment of Radar in real-time scene

The file wave_comparison.py can be used to dispaly the original vibraion wave, the simulation wave fitted based on the original one and the vibration waves used as control sample.

#IMU: SparkFun DataLogger IoT - 9DoF

#Motor driver: SparkFun Haptic Motor Driver - DRV2605L

#Microcontroller: SparkFun Thing Plus - ESP32 WROOM (Micro-B)

#Motor: 1020-15-003-001 (ERM Motor)

The internet platform to receive HTTP: https://thingspeak.com    (MATLAB account required)

Notes:

Even if the voltage for output of driver is set as 0, the motor won't stop working.
The GPIO35, GPIO36, GPIO39, GPIO34 pins on SparkFun Thing Plus - ESP32 WROOM (Micro-B) are invalid to be set as output pin if the push button connects with the board. However, it is available without push button.

