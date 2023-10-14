// SparkFun DataLogger IoT – 9DoF Test Example
// Tested with Espressif ESP32 v2.0.5 and the "ESP32 Dev Module" board definition

/**********************************************************************************************
 *
 * WARNING!
 * 
 * This is a sketch we wrote to test the DataLogger IoT – 9DoF hardware.
 * Please think very carefully before uploading it to your DataLogger.
 * 
 * You will overwrite the DataLogger firmware, leaving it unable to update or restore itself. 
 * 
 * The DataLogger IoT – 9DoF comes pre-programmed with amazing firmware which can do _so_ much.
 * It is designed to be able to update itself and restore itself if necessary.
 * But it can not do that if you overwrite the firmware with this test sketch.
 * It is just like erasing the restore partition on your computer hard drive.
 * Do not do it - unless you really know what you are doing.
 * 
 * Really. We mean it.
 * 
 * Your friends at SparkFun.
 * 
 * License: MIT. Please see LICENSE.MD for more details
 * 
 **********************************************************************************************/

// Based on: https://github.com/espressif/arduino-esp32/blob/master/libraries/SD_MMC/examples/SDMMC_Test/SDMMC_Test.ino

#define EN_3V3_SW 32 // The 3.3V_SW regulator Enable pin is connected to D32

#include "FS.h"
#include "SD_MMC.h"
//#define IMU_CS SS // The ISM330 chip select is connected to D5
uint8_t IMU_CS = 5;
#include <SPI.h>
#include "SparkFun_ISM330DHCX.h" //Click here to get the library: http://librarymanager/All#SparkFun_6DoF_ISM330DHCX
#include <Wire.h>
//#include <RTClib.h> // without ms
#include <DS3231.h> //including ms

// SPI instance class call
SparkFun_ISM330DHCX_SPI myISM; 

RTClib myRTC;  // Create an instance of the DS3231 RTC

unsigned long lastMillis = 0; // Variable to store the last time


//DateTime now; // Create an instance of the RTCDateTime class to hold the date and time

// Structs for X,Y,Z data
sfe_ism_data_t accelData; 
sfe_ism_data_t gyroData; 


void listDir(fs::FS &fs, const char * dirname, uint8_t levels){
    Serial.printf("Listing directory: %s\n", dirname);

    File root = fs.open(dirname);
    if(!root){
        Serial.println("Failed to open directory");
        return;
    }
    if(!root.isDirectory()){
        Serial.println("Not a directory");
        return;
    }

    File file = root.openNextFile();
    while(file){
        if(file.isDirectory()){
            Serial.print("  DIR : ");
            Serial.println(file.name());
            if(levels){
                listDir(fs, file.path(), levels -1);
            }
        } else {
            Serial.print("  FILE: ");
            Serial.print(file.name());
            Serial.print("  SIZE: ");
            Serial.println(file.size());
        }
        file = root.openNextFile();
    }
}

void createDir(fs::FS &fs, const char * path){
    Serial.printf("Creating Dir: %s\n", path);
    if(fs.mkdir(path)){
        Serial.println("Dir created");
    } else {
        Serial.println("mkdir failed");
    }
}

void removeDir(fs::FS &fs, const char * path){
    Serial.printf("Removing Dir: %s\n", path);
    if(fs.rmdir(path)){
        Serial.println("Dir removed");
    } else {
        Serial.println("rmdir failed");
    }
}

void readFile(fs::FS &fs, const char * path){
    Serial.printf("Reading file: %s\n", path);

    File file = fs.open(path);
    if(!file){
        Serial.println("Failed to open file for reading");
        return;
    }

    Serial.print("Read from file: ");
    while(file.available()){
        Serial.write(file.read());
    }
}

void writeFile(fs::FS &fs, const char * path, const char * message){
    Serial.printf("Writing file: %s\n", path);

    File file = fs.open(path, FILE_WRITE);
    if(!file){
        Serial.println("Failed to open file for writing");
        return;
    }
    if(file.print(message)){
        Serial.println("File written");
    } else {
        Serial.println("Write failed");
    }
}

void appendFile(fs::FS &fs, const char * path, const char * message){
    Serial.printf("Appending to file: %s\n", path);

    File file = fs.open(path, FILE_APPEND);
    if(!file){
        Serial.println("Failed to open file for appending");
        return;
    }
    if(file.print(message)){
        Serial.println("Message appended");
    } else {
        Serial.println("Append failed");
    }
}

void renameFile(fs::FS &fs, const char * path1, const char * path2){
    Serial.printf("Renaming file %s to %s\n", path1, path2);
    if (fs.rename(path1, path2)) {
        Serial.println("File renamed");
    } else {
        Serial.println("Rename failed");
    }
}

void deleteFile(fs::FS &fs, const char * path){
    Serial.printf("Deleting file: %s\n", path);
    if(fs.remove(path)){
        Serial.println("File deleted");
    } else {
        Serial.println("Delete failed");
    }
}

void testFileIO(fs::FS &fs, const char * path){
    File file = fs.open(path);
    static uint8_t buf[512];
    size_t len = 0;
    uint32_t start = millis();
    uint32_t end = start;
    if(file){
        len = file.size();
        size_t flen = len;
        start = millis();
        while(len){
            size_t toRead = len;
            if(toRead > 512){
                toRead = 512;
            }
            file.read(buf, toRead);
            len -= toRead;
        }
        end = millis() - start;
        Serial.printf("%u bytes read for %u ms\n", flen, end);
        file.close();
    } else {
        Serial.println("Failed to open file for reading");
    }


    file = fs.open(path, FILE_WRITE);
    if(!file){
        Serial.println("Failed to open file for writing");
        return;
    }

    size_t i;
    start = millis();
    for(i=0; i<2048; i++){
        file.write(buf, 512);
    }
    end = millis() - start;
    Serial.printf("%u bytes written for %u ms\n", 2048 * 512, end);
    file.close();
}
void writeAccelerometerDataToCSV() {
    // Format accelerometer data as a CSV string
    //String dataString = String(accelData.xData) + "," + String(accelData.yData) + "," + String(accelData.zData) + "\n";

    // Get the current time from the RTC
    DateTime now = myRTC.now();

        // Measure milliseconds since last second
    //unsigned long currentMillis = millis() % 1000;

    unsigned long currentMillis = millis();

    // Calculate seconds, minutes, and hours
    unsigned long milliseconds = currentMillis % 1000;
    unsigned long seconds = (currentMillis / 1000) % 60;
    unsigned long minutes = (currentMillis / (1000 * 60)) % 60;
    unsigned long hours = (currentMillis / (1000 * 60 * 60)) % 24;

    // Format timestamp, accelerometer data, and time as a CSV string
    String dataString = String(hours) + ":" + String(minutes) + ":" + String(seconds) + "." + String(milliseconds) + "," +
                        String(accelData.xData) + "," + String(accelData.yData) + "," + String(accelData.zData) + "\n";
    // Specify the file path
    const char *filePath = "/Mydirectory/vibration.CSV";

    // Check if the file exists
    if (!SD_MMC.exists(filePath)) {
        // If the file doesn't exist, create it and add a header line
        const char *header = "Timestamp,X,Y,Z\n";
        writeFile(SD_MMC, filePath, header);
    }

    // Write the formatted accelerometer data to the CSV file
    appendFile(SD_MMC, filePath, dataString.c_str());
}
void setup()
{

  pinMode(EN_3V3_SW, OUTPUT); // Enable power for the microSD card
  digitalWrite(EN_3V3_SW, HIGH);
  pinMode(IMU_CS, OUTPUT);
	digitalWrite(IMU_CS, HIGH);

  delay(1000); // Allow time for the SD card to start up

  SPI.begin();
  Serial.begin(115200);



  if( !myISM.begin(IMU_CS) ){
	  Serial.println(F("IMU did not begin. Freezing..."));
	  while(1);
	}
  // Reset the device to default settings
	// This if helpful is you're doing multiple uploads testing different settings. 
	myISM.deviceReset();

	// Wait for it to finish reseting
	while( !myISM.getDeviceReset() ){ 
		delay(1);
	} 

	Serial.println(F("IMU has been reset."));
	Serial.println(F("Applying settings..."));
	delay(100);
	
	myISM.setDeviceConfig();
	myISM.setBlockDataUpdate();
	
	// Set the output data rate and precision of the accelerometer
	myISM.setAccelDataRate(ISM_XL_ODR_104Hz);
	myISM.setAccelFullScale(ISM_4g); 

	// Set the output data rate and precision of the gyroscope
	myISM.setGyroDataRate(ISM_GY_ODR_104Hz);
	myISM.setGyroFullScale(ISM_500dps); 

	// Turn on the accelerometer's filter and apply settings. 
	myISM.setAccelFilterLP2();
	myISM.setAccelSlopeFilter(ISM_LP_ODR_DIV_100);

	// Turn on the gyroscope's filter and apply settings. 
	myISM.setGyroFilterLP1();
	myISM.setGyroLP1Bandwidth(ISM_MEDIUM);

  if(!SD_MMC.begin()){
      Serial.println("Card Mount Failed");
      return;
  }

  uint8_t cardType = SD_MMC.cardType();

  if(cardType == CARD_NONE){
      Serial.println("No SD_MMC card attached");
      return;
  }

  Serial.print("SD_MMC Card Type: ");
  if(cardType == CARD_MMC){
      Serial.println("MMC");
  } else if(cardType == CARD_SD){
      Serial.println("SDSC");
  } else if(cardType == CARD_SDHC){
      Serial.println("SDHC");
  } else {
      Serial.println("UNKNOWN");
  }

  uint64_t cardSize = SD_MMC.cardSize() / (1024 * 1024);
  Serial.printf("SD_MMC Card Size: %lluMB\n", cardSize);

  // Create a directory named "myDirectory"
  listDir(SD_MMC, "/", 0);
  createDir(SD_MMC, "/Mydirectory");


  //listDir(SD_MMC, "/", 0);
  //createDir(SD_MMC, "/mydir");
  //listDir(SD_MMC, "/", 0);
  //removeDir(SD_MMC, "/mydir");
  //listDir(SD_MMC, "/", 2);
  //writeFile(SD_MMC, "/hello.CSV", &accelData);
  //appendFile(SD_MMC, "/hello.txt", "World!\n");
  //readFile(SD_MMC, "/hello.txt");
  //deleteFile(SD_MMC, "/foo.txt");
  //renameFile(SD_MMC, "/hello.txt", "/foo.txt");
  //readFile(SD_MMC, "/foo.txt");
  //testFileIO(SD_MMC, "/test.txt");
  Serial.printf("Total space: %lluMB\n", SD_MMC.totalBytes() / (1024 * 1024));
  Serial.printf("Used space: %lluMB\n", SD_MMC.usedBytes() / (1024 * 1024));
}

void loop()
{
  	if( myISM.checkStatus() ){
		myISM.getAccel(&accelData);

    // Get the current time from the RTC
    DateTime now = myRTC.now();
        // Measure milliseconds since last second
    //unsigned long currentMillis = millis() % 1000;

    unsigned long currentMillis = millis();

    // Calculate seconds, minutes, and hours
    unsigned long milliseconds = currentMillis % 1000;
    unsigned long seconds = (currentMillis / 1000) % 60;
    unsigned long minutes = (currentMillis / (1000 * 60)) % 60;
    unsigned long hours = (currentMillis / (1000 * 60 * 60)) % 24;

    // Print the time and accelerometer data to the serial monitor
    //Serial.print(rtc.getDate);
    //Serial.print("--");
    //Serial.print(rtc.getTime);
   // Serial.print(":");
    //Serial.print(now.second());
  // Print the date and time

    Serial.print(hours);
    Serial.print(':');
    Serial.print(minutes);
    Serial.print(':');
    Serial.print(seconds);
    Serial.println('.');
    Serial.println(milliseconds);
    Serial.print(F(","));
		//myISM.getGyro(&gyroData);
		//Serial.print(F("Accelerometer: "));
		//Serial.print(F("X: "));
		Serial.print(accelData.xData);
		Serial.print(F(","));
		//Serial.print(F("Y: "));
		Serial.print(accelData.yData);
		Serial.print(F(","));
		//Serial.print(F("Z: "));
		Serial.print(accelData.zData);
		Serial.println(F(","));
    // Write accelerometer data to a CSV file
    writeAccelerometerDataToCSV();

    delay(20);
    }
    else {
    Serial.println("Failed to read IMU data.");
    }
  //delay(1000); 
}