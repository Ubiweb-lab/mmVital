/*
  Grab a Timestamp from the most recent button press/EVI pin event
  By: Andy England
  SparkFun Electronics
  Date: 2/22/2017
  License: This code is public domain but you buy me a beer if you use this and we meet someday (Beerware license).

  Feel like supporting our work? Buy a board from SparkFun!
  https://www.sparkfun.com/products/14642

  This example shows how to get the timestamp of an event generated on the EVI pin, either by a button press or an MCU generated event

  Hardware Connections:
    Plug the RTC into the Qwiic port on your microcontroller or on your Qwiic shield/adapter.
    If you are using an adapter cable, here is the wire color scheme: 
    Black=GND, Red=3.3V, Blue=SDA, Yellow=SCL
    Open the serial monitor at 115200 baud
*/

#include <SparkFun_RV8803.h> //Get the library here:http://librarymanager/All#SparkFun_RV-8803

RV8803 rtc;

void setup()
{
  Wire.begin();

  Serial.begin(115200);
  Serial.println("Alarm from RTC Example");

  if (rtc.begin() == false)
  {
    Serial.println("Device not found. Please check wiring. Freezing.");
    while(1);
  }
  Serial.println("RTC online!");
  
  rtc.setEVIEventCapture(RV8803_ENABLE); //Enables the Timestamping function
  rtc.setEVIDebounceTime(EVI_DEBOUNCE_256HZ); //Debounce the button with a 3.9 ms(256 Hz) sampling period, other options are EVI_DEBOUNCE_NONE, EVI_DEBOUNCE_64HZ, and EVI_DEBOUNCE_8HZ
  //rtc.setEVIEdgeDetection(RISING_EDGE); // Uncomment to set event detection to button release instead of press
}

void loop()
{
  if (rtc.getInterruptFlag(FLAG_EVI))
  {
    rtc.updateTime();
    rtc.clearInterruptFlag(FLAG_EVI);

    String currentDate = rtc.stringDateUSA(); //Get the current date in mm/dd/yyyy format (we're weird)
    //String currentDate = rtc.stringDate(); //Uncomment this line to get the current date in dd/mm/yyyy format
    String timestamp = rtc.stringTimestamp();

    Serial.print(currentDate);
    Serial.print(" ");
    Serial.println(timestamp);
  }
}