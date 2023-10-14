#include <Sparkfun_DRV2605L.h>
#include <Wire.h>
#include <Gaussian.h>

SFE_HMD_DRV2605L HMD;

const int F = 1;          // Frequency of the signal
const int Fs = 50;        // Sampling frequency
const int n = 50;         // Number of samples
const int analogOutPin1 = 8; // Analog output pin that the Haptic Motor Driver is attached to

// Noise parameters
const float noise_mean = 0.0;  // Mean of the Gaussian noise (usually 0)
const float noise_stddev = 10.0; // Standard deviation of the Gaussian noise (adjust as needed)

float generateColoredGaussianNoise(float mean, float stddev, float alpha, float Ts) {
  static float z1 = 0.0;  // Internal state for the filter
  float z0 = sqrt(-2 * log(static_cast<float>(rand()) / RAND_MAX)) * cos(2 * M_PI * static_cast<float>(rand()) / RAND_MAX);
  float noise = mean + sqrt(alpha) * z0 + (1 - alpha) * z1;
  z1 = z0;
  return stddev * noise;
}

void setup() {
  HMD.begin();
  Serial.begin(9600);
  HMD.Mode(0x03); // PWM INPUT 
  HMD.MotorSelect(0x0A);
  HMD.Library(7); // Change to 6 for LRA motors 

  pinMode(analogOutPin1, OUTPUT);

  float alpha = 0.95;  // Parameter for the colored noise (0 <= alpha < 1)

  for (int i = 0; i < n; i++) {
    float t = static_cast<float>(i) / Fs;
    float sinValue = 127.0 * sin(2 * M_PI * F * t) + 127.0;
    float noise = generateColoredGaussianNoise(noise_mean, noise_stddev, alpha, 1.0 / Fs);
    byte sample = static_cast<byte>(sinValue + noise);
    analogWrite(analogOutPin1, sample);
    delayMicroseconds(1000000 / Fs);
    Serial.print("\t output_Sinvalue = ");
    Serial.println(sinValue);
    Serial.print("\t output = ");
    Serial.println(sample);
  }
}

void loop() {
  // No action in the loop for this example
}

