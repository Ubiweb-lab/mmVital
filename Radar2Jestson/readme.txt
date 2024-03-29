DCA1000 Captures LVDS data from IWR6843ISK radar sensors
Streams output in real time through 1-Gbps Ethernet (from DCA1000 to Jetson Nano)

mmWave software development kit (SDK) for linux (Jetson)
Download: https://www.ti.com/tool/MMWAVE-SDK

mmWave Demo Visualizer (windows, Mac and Linux): 
https://github.com/sgs-weather-and-environmental-systems/demo-visualizer
This version of app is compatible with mmWave SDK versions 1.2.0, 1.1.0 and 1.0.0.

DCA1000 configuration in the file: datacard_config.json

How to get DCA1000 CLI Utility for Linux
The Linux based CLI sourcecode can be obtained from mmWave Studio package. 
Please download and install mmWave Studio (MMWAVE-STUDIO) version 2.1 or later from http://www.ti.com/tool/MMWAVE-STUDIO and browse to the installed folder: C:\ti\mmwave_studio_<ver>\mmWaveStudio\ReferenceCode\DCA1000. 
Copy the 'SourceCode' folder to a linux machine and use the instructions from C:\ti\mmwave_studio_<ver>\mmWaveStudio\ReferenceCode\DCA1000\Docs\TI_DCA1000EVM_CLI_Software_UserGuide. pdf to build the utility for your linux distribution.

cd /home/wobble/Radar2Jetson
make
cp configfile.json /home/wobble/Radar2Jetson/DCA1000/SourceCode/Release/

After Radar and DCA1000 are connect with Jetson
configure the network eth0 as 192.168.33.30
$lsusb #show the prot

DCA1000EVM CLI Setup
➢ Copy the DCA1000EVM CLI binaries in the PC. Refer section 3.1.2 and 3.2.2 for the list of CLI binaries required for Windows and Linux platforms respectively.
➢ DCA1000EVM should be connected to Host PC via Ethernet cable to access the CLI and Data Transfer process.
➢ DCA1000EVM should be connected to PC via USB Cable (J1-Radar FTDI) for configuring the RADAR EVM by using on board FTDI chip.
➢ DCA1000EVM should be connected to TI Radar EVM via 60 pin HD Connector by using 60 pin Samtec ribbon cable.
➢ DCA1000EVM power input should be connected either from DC Jack or TI Radar EVM power output (from 60 pin HD connector) by selecting the switch SW3.
➢ RADAR EVM should be connected to ±5V power supply.
➢ Follow the mmWave Studio or mmWave SDK User Guide for additional RADAR EVM connectivity to PC and
other pre-requisites.

Files Requirements To Execute CLI
Files required for CLI execution:
Files                                       Description
DCA1000EVM_CLI_Control                      Executable file that does validation of user inputs and execution of configuration commands
DCA1000EVM_CLI_Record                       Executable file that does validation of user inputs and execution of record commands
libRF_API.so                                Dynamic library file that handles execution of configuration commands and recording of the  
                                            captured data in files
configFile.json                             JSON file for configuration from user

➢ Open the command prompt and move to the directory where the above-mentioned files are downloaded.
➢ Make the files DCA1000EVM_CLI_Control and DCA1000EVM_CLI_Record as executables using ‘sudo chmod +x DCA1000EVM_CLI_Control’ and ‘sudo chmod +x DCA1000EVM_CLI_Record’ commands (if the files are
not already in the executable mode).
➢ Update LD_LIBRARY_PATH using 

Option 1: ‘export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$pwd’ command.

Option 2:
nano ~/.bashrc
Add the following line to the end of the file:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$pwd
Control+x
source ~/.bashrc



Start recording using command sequence:

CLI Command Sequence
For successful recording of data from RADAR EVM sequence is given as follows
1. Configure FPGA
▪ Ensure JSON config file (CLI) and Script config file (RADAR EVM) data format mode are in sync ▪ Command - ./DCA1000EVM_CLI_Control fpga configFile.json
2. Configure record delay
▪ Command - ./DCA1000EVM_CLI_Control record configFile.json
3. Start the record
▪ Ensure JSON config file (CLI) and Script config file (RADAR EVM) data logging mode are in sync ▪ Command - ./DCA1000EVM_CLI_Control start_record configFile.json
36 DCA1000EVM CLI Software User Guide 1.01 Copyright @2019. Texas Instruments Incorporated
 
 www.ti.com DCA1000EVM CLI Execution Instructions 4. Stop the record after recording data
▪ Command - ./DCA1000EVM_CLI_Control stop_record configFile.json For successful record process, FPGA should be reconfigured in the following scenarios
➢ When the system is booted or rebooted
➢ When the FPGA or DCA1000EVM is reset
➢ On switching between multi-mode and raw mode

For the "raw" mode, the data filename would be <File_Prefix>_Raw_<iteration>.bin. Raw mode is only supported for H/W session and the saved data is directly payload with no header or extra bytes/padding


Serial Communication Library: 
pip install pyserial

Sample python scripts to parse the cloud point output of the demo are provided in mmwave_sdk_<ver>\packages\ti\demo\parser_scripts

---------------------------------------------------------------------------------------------
Ensure the AWR1642 device has pre-flashed image in it, if not users need to use TI Uniflash tool ( http://www.ti.com/tool/UNIFLASH) to flash the device. SOP2 jumper and SOP0 jumper should be connected for starting Flash Programming Mode. The image in the directory of mmwave_SDK like mmwave_sdk_<ver>\ti\demo\<platform>\mmw\<platform>_mmw_demo.bin. 

Radar_configuration.py can be used to start and stop AWR1642 millimeter Radar.
Notes: Users need to switch the 'serial_port_CLI' to their own port in this Python file. 
The command './DCA1000EVM_CLI_Control start_record configFile.json' can be run to start record on DCA1000 after the Radar is working.

Extension of large space NVMe storage on the Jetson-Orin-Nano/NX, users need to use physical PC with Ubuntu system rather than virtual system, and then download the SDK manager (https://docs.nvidia.com/sdk-manager/download-run-sdkm/index.html) to flash the Jetson-Orin-Nano/NX.
