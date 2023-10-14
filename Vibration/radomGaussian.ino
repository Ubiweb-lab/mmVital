#include <Sparkfun_DRV2605L.h>
#include <Wire.h>
#include <Gaussian.h>

SFE_HMD_DRV2605L HMD;

int F = 1;          // Frequency of the signal
int Fs = 50;        // Sampling frequency
int n = 50;         // Number of samples
float t;            // Time instance
int sampling_interval;
byte samples[50];   // To store the samples
byte sinValue[50];
const int analogOutPin1 = 15; // Analog output pin that the Haptic Motor Driver is attached to

// Noise parameters
float noise_mean = 0.0; // Mean of Gaussian noise (usually 0)
float noise_stddev = 10.0; // Standard deviation of Gaussian noise (adjust as needed)

  
float generateGaussianNoise(float mean, float stddev) {
  float u1 = static_cast<float>(rand()) / RAND_MAX; // Generate a random number in [0, 1]
  float u2 = static_cast<float>(rand()) / RAND_MAX; // Generate a random number in [0, 1]

  // Box-Muller transform to generate two independent standard normal variables
  float z0 = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);

  // Scale and shift the generated standard normal variable to have the desired mean and standard deviation
  return mean + z0 * stddev;
}

void setup() {
  // Initialize Haptic Motor Driver and other settings
  HMD.begin();
  Serial.begin(9600);
  HMD.Mode(0x03); // PWM INPUT 
  HMD.MotorSelect(0x0A);
  HMD.Library(7); // Change to 6 for LRA motors 

  pinMode(analogOutPin1, OUTPUT);

  // Generate the sine wave signal with added Gaussian noise
  for (int n = 0; n < 50; n++) {
    t = (float)n / Fs;
    sinValue[n] = (byte) (127.0 * sin(2 * 3.14 * t) + 127.0);
    //sinValue[n] = (byte) (4.01 * sin(0.97 * t) + 111.86);
    float noise = generateGaussianNoise(noise_mean, noise_stddev);
    samples[n] = (byte)(sinValue[n] + noise);
  }

  // Calculate the sampling interval
  sampling_interval = 1000000 / (F * n); // Sampling interval Ts = 1/frequency x number of samples (Ts = 1/Fn or Ts = T/n) x 1000000 to convert it to microseconds
}

void loop() {
  for (int j = 0; j < 50; j++) {
    analogWrite(analogOutPin1, samples[j]);
    delayMicroseconds(sampling_interval); // Time interval
    Serial.print("\t output_Sinvalue = ");
    Serial.println(sinValue[j]);
    Serial.print("\t output = ");
    Serial.println(samples[j]);
  }
}