#include <Sparkfun_DRV2605L.h>
#include <Wire.h>

SFE_HMD_DRV2605L HMD;

int buttonPin = 26;  // The pin your button is connected to


const int analogOutPin0 = 14; // Analog output pin that the Haptic Motor Driver is attached to
const int analogOutPin1 = 32; // Analog output pin that the Haptic Motor Driver is attached to
const int analogOutPin2 = 15; // Analog output pin that the Haptic Motor Driver is attached to
const int analogOutPin3 = 33;
const int analogOutPin4 = 04; //35
const int analogOutPin5 = 16; //36
const int analogOutPin6 = 19; //39
const int analogOutPin7 = 23; //35

int sample_0 = 255;
int sample_1 = 0;
int sample_2 = 100;
//int sample_3 = 250;

int currentSample = sample_0;

int oldButtonState = LOW;

void setup() {
  HMD.begin();
  Serial.begin(115200);
  HMD.Mode(0x03); // PWM INPUT 
  HMD.MotorSelect(0x0A);
  HMD.Library(7); // Change to 6 for LRA motors

  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(analogOutPin0, OUTPUT);
  pinMode(analogOutPin1, OUTPUT);
  pinMode(analogOutPin2, OUTPUT);
  pinMode(analogOutPin3, OUTPUT);
  pinMode(analogOutPin4, OUTPUT);
  pinMode(analogOutPin5, OUTPUT);
  pinMode(analogOutPin6, OUTPUT);
  pinMode(analogOutPin7, OUTPUT);


  //analogWrite(analogOutPin5, 255);
  //analogWrite(analogOutPin5, 0);  // Start with the output off
}

void loop() {
  int buttonState = digitalRead(buttonPin);

  if (buttonState != oldButtonState && buttonState == HIGH) {
    // Button is pressed, switch the analog value
    analogWrite(analogOutPin0, currentSample);
    analogWrite(analogOutPin1, currentSample);
    analogWrite(analogOutPin2, currentSample);
    analogWrite(analogOutPin3, currentSample);
    analogWrite(analogOutPin4, currentSample);
    analogWrite(analogOutPin5, currentSample);
    analogWrite(analogOutPin6, currentSample);
    analogWrite(analogOutPin7, currentSample);  // Set the current sample

    // Cycle through the samples
    if (currentSample == sample_0)
      currentSample = sample_1;
    else if (currentSample == sample_1)
      currentSample = sample_2;
    //else if (currentSample == sample_2)
    //  currentSample = sample_3;
    else
      currentSample = sample_0;
  }

  oldButtonState = buttonState;
  delay(50);
}
