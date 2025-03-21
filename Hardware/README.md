# SPARK Hardware
BOM

## CAD
The [CAD](./CAD/) folder contains the [FreeCAD](https://snapcraft.io/freecad-realthunder) sorce files, along with *.3mf* files for 3D printing. 

## PCB
To assemble the electronics, fabricate the PCBs found in the [PCB](./PCB/) folder. The components can be found in the BOM. 

## Firmware
This project uses PlatformIO projects found in the [Firmware](./Firmware/) folder. The recommended method is to use the [PlatformIO](https://docs.platformio.org/en/latest/integration/ide/vscode.html) extension for VSCode.       

The electronics for this project uses one [ESP8266 NodeMCU microcontroller](./Firmware/SparkSerialTX/) for each SPARK unit, and one [STM32F103C8T6 Blue Pill](./Firmware/Motor_controller/) board for the haptic gloves. 
