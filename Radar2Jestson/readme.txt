DCA1000 Captures LVDS data from IWR6843ISK radar sensors
Streams output in real time through 1-Gbps Ethernet (from DCA1000 to Jetson Nano)

mmWave software development kit (SDK) for linux (Jetson)
Download: https://www.ti.com/tool/MMWAVE-SDK

DCA1000 configuration in the file: datacard_config.json

How to get DCA1000 CLI Utility for Linux
The Linux based CLI sourcecode can be obtained from mmWave Studio package. 
Please download and install mmWave Studio (MMWAVE-STUDIO) version 2.1 or later from http://www.ti.com/tool/MMWAVE-STUDIO and browse to the installed folder: C:\ti\mmwave_studio_<ver>\mmWaveStudio\ReferenceCode\DCA1000. 
Copy the 'SourceCode' folder to a linux machine and use the instructions from C:\ti\mmwave_studio_<ver>\mmWaveStudio\ReferenceCode\DCA1000\Docs\TI_DCA1000EVM_CLI_Software_UserGuide. pdf to build the utility for your linux distribution.


Linux::DCA1000EVM CLI Commands

# configure DCA1000EVM
./DCA1000EVM_CLI_Control fpga datacard_config.json
#configure CLI application with the record related settings
./DCA1000EVM_CLI_Control record datacard_config.json
#start record and wait for the data over ethernet
./DCA1000EVM_CLI_Control start_record datacard_config.json

Linux::Stop DCA1000EVM_CLI

./DCA1000EVM_CLI_Control stop_record datacard_config.json

Serial Communication Library: 
pip install pyserial

Sample python scripts to parse the cloud point output of the demo are provided in mmwave_sdk_<ver>\packages\ti\demo\parser_scripts
