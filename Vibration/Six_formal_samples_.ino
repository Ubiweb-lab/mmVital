#include <Sparkfun_DRV2605L.h>
#include <Wire.h>

SFE_HMD_DRV2605L HMD;

int buttonPin = 26;  // The pin your button is connected to

//int currentSample = sample_0;
int currentSampleIndex = 0; // Current index of the active sample

int oldButtonState = LOW;



const int analogOutPins[] = {14, 32, 15, 33, 4, 17, 16, 19};
const int numPins = 8;


const int numSamples = 1000;
const float sampleRate = 1000;
const float frequency = 1.0;
const float noise_mean = 0.0;
const float noise_stddev = 10.0;

byte samples_0[numSamples];
byte samples_1[numSamples];
byte samples_2[numSamples];
byte samples_3[numSamples];

byte *activeSample;


// Define alpha and Fs
const float alpha = 0.5;  // Adjust this value accordingly
const float Fs = 1000.0;  // Adjust this value accordingly



float generateColoredGaussianNoise(float mean, float stddev, float alpha, float Ts) {
  static float z1 = 0.0;  // Internal state for the filter
  float z0 = sqrt(-2 * log(static_cast<float>(rand()) / RAND_MAX)) * cos(2 * M_PI * static_cast<float>(rand()) / RAND_MAX);
  float noise = mean + sqrt(alpha) * z0 + (1 - alpha) * z1;
  z1 = z0;
  return stddev * noise;
}

void generateSample0() {
  for (int i = 0; i < numSamples; i++) {
    float t = i / sampleRate;
    float sampleValue = 1084.9 * sin(2 * PI * frequency * t) / 32767 * 255;
    samples_0[i] = (byte)sampleValue;
        // Print the sample value
    Serial.print(samples_0[i]);
    Serial.print(", ");
  }
}

void generateSample1() {
  for (int i = 0; i < numSamples; i++) {
    float t = i / sampleRate;
    float sampleValue = 1084.9 * sin(2 * PI * frequency * t);
    float noiseValue = generateColoredGaussianNoise(noise_mean, noise_stddev, alpha, 1.0 / Fs);
    samples_1[i] = (byte)(sampleValue + noiseValue / 32767 * 255);
        // Print the sample value
    Serial.print(samples_1[i]);
    Serial.print(", ");
  }
}

void generateSample2() {
  for (int i = 0; i < numSamples; i++) {
    float t = i / sampleRate;
    samples_2[i] = (byte)((1021.75 - 0.08 * cos(2 * PI * 1 * t) - 0.00 * sin(2 * PI * 1 * t) +
                         0.04 * cos(4 * PI * 1 * t) + 0.01 * sin(4 * PI * 1 * t) +
                         0.05 * cos(6 * PI * 1 * t) + 0.04 * sin(6 * PI * 1 * t) +
                         -0.01 * cos(8 * PI * 1 * t) - 0.06 * sin(8 * PI * 1 * t) +
                         -0.02 * cos(10 * PI * 1 * t) + 0.67 * sin(10 * PI * 1 * t) +
                         0.23 * cos(12 * PI * 1 * t) - 0.12 * sin(12 * PI * 1 * t) +
                         -0.03 * cos(14 * PI * 1 * t) + 0.02 * sin(14 * PI * 1 * t) +
                         -0.01 * cos(16 * PI * 1 * t) - 0.01 * sin(16 * PI * 1 * t) +
                         0.02 * cos(18 * PI * 1 * t) - 0.01 * sin(18 * PI * 1 * t) +
                         -0.01 * cos(20 * PI * 1 * t) - 0.00 * sin(20 * PI * 1 * t) +
                         0.03 * cos(22 * PI * 1 * t) + 0.07 * sin(22 * PI * 1 * t) +
                         -0.13 * cos(24 * PI * 1 * t) - 3.21 * sin(24 * PI * 1 * t) +
                         -2.13 * cos(26 * PI * 1 * t) + 0.07 * sin(26 * PI * 1 * t) +
                         0.10 * cos(28 * PI * 1 * t) + 0.01 * sin(28 * PI * 1 * t) +
                         -0.01 * cos(30 * PI * 1 * t) + 0.03 * sin(30 * PI * 1 * t) +
                         0.01 * cos(32 * PI * 1 * t) - 0.00 * sin(32 * PI * 1 * t) +
                         0.05 * cos(34 * PI * 1 * t) - 0.02 * sin(34 * PI * 1 * t) +
                         0.00 * cos(36 * PI * 1 * t) + 0.14 * sin(36 * PI * 1 * t) +
                         -0.08 * cos(38 * PI * 1 * t) + 1.09 * sin(38 * PI * 1 * t) +
                         -0.54 * cos(40 * PI * 1 * t) - 0.03 * sin(40 * PI * 1 * t) +
                         -0.08 * cos(42 * PI * 1 * t) + 0.04 * sin(39* 2 * PI * 1 * t) + 
                         -0.00 * cos(40* 2 * PI * 1 * t) + -0.01 * sin(40* 2 * PI * 1 * t) + 
                         -0.04 * cos(41* 2 * PI * 1 * t) + 0.03 * sin(41* 2 * PI * 1 * t) + 
                         -0.02 * cos(42* 2 * PI * 1 * t) + 0.03 * sin(42* 2 * PI * 1 * t) + 
                         -0.04 * cos(43* 2 * PI * 1 * t) + -0.19 * sin(43* 2 * PI * 1 * t) + 
                         1.14 * cos(44* 2 * PI * 1 * t) + 0.02 * sin(44* 2 * PI * 1 * t) + 
                         -0.04 * cos(45* 2 * PI * 1 * t) + -0.03 * sin(45* 2 * PI * 1 * t) + 
                         0.00 * cos(46* 2 * PI * 1 * t) + -0.02 * sin(46* 2 * PI * 1 * t) + 
                         -0.03 * cos(47* 2 * PI * 1 * t) + 0.03 * sin(47* 2 * PI * 1 * t) + 
                         0.00 * cos(48* 2 * PI * 1 * t) + 0.06 * sin(48* 2 * PI * 1 * t) + 
                         -0.05 * cos(49* 2 * PI * 1 * t) + -0.44 * sin(49* 2 * PI * 1 * t) + 
                         0.45 * cos(50* 2 * PI * 1 * t) + -0.09 * sin(50* 2 * PI * 1 * t) + 
                         -0.03 * cos(51* 2 * PI * 1 * t) + 0.04 * sin(51* 2 * PI * 1 * t) + 
                         0.04 * cos(52* 2 * PI * 1 * t) + 0.01 * sin(52* 2 * PI * 1 * t) + 
                         -0.00 * cos(53* 2 * PI * 1 * t) + 0.02 * sin(53* 2 * PI * 1 * t) + 
                         -0.05 * cos(54* 2 * PI * 1 * t) + 0.07 * sin(54* 2 * PI * 1 * t) + 
                         0.04 * cos(55* 2 * PI * 1 * t) + 0.02 * sin(55* 2 * PI * 1 * t) + 
                         -0.05 * cos(56* 2 * PI * 1 * t) + 0.04 * sin(56* 2 * PI * 1 * t) + 
                         0.19 * cos(57* 2 * PI * 1 * t) + 0.05 * sin(57* 2 * PI * 1 * t) + 
                         0.03 * cos(58* 2 * PI * 1 * t) + 0.00 * sin(58* 2 * PI * 1 * t) + 
                         0.03 * cos(59* 2 * PI * 1 * t) + -0.02 * sin(59* 2 * PI * 1 * t) + 
                         -0.01 * cos(60* 2 * PI * 1 * t) + -0.01 * sin(60* 2 * PI * 1 * t) + 
                         -0.05 * cos(61* 2 * PI * 1 * t) + -0.08 * sin(61* 2 * PI * 1 * t) + 
                         -0.10 * cos(62* 2 * PI * 1 * t) + 2.16 * sin(62* 2 * PI * 1 * t) + 
                         0.86 * cos(63* 2 * PI * 1 * t) + -0.00 * sin(63* 2 * PI * 1 * t) + 
                         -0.04 * cos(64* 2 * PI * 1 * t) + 0.00 * sin(64* 2 * PI * 1 * t) + 
                         0.00 * cos(65* 2 * PI * 1 * t) + -0.03 * sin(65* 2 * PI * 1 * t) + 
                         0.03 * cos(66* 2 * PI * 1 * t) + -0.01 * sin(66* 2 * PI * 1 * t) + 
                         -0.02 * cos(67* 2 * PI * 1 * t) + 0.08 * sin(67* 2 * PI * 1 * t) + 
                         0.04 * cos(68* 2 * PI * 1 * t) + -0.04 * sin(68* 2 * PI * 1 * t) + 
                         -2.49 * cos(69* 2 * PI * 1 * t) + 0.01 * sin(69* 2 * PI * 1 * t) + 
                         -0.00 * cos(70* 2 * PI * 1 * t) + -0.00 * sin(70* 2 * PI * 1 * t) + 
                         -0.04 * cos(71* 2 * PI * 1 * t) + 0.03 * sin(71* 2 * PI * 1 * t) + 
                         0.03 * cos(72* 2 * PI * 1 * t) + -0.02 * sin(72* 2 * PI * 1 * t) + 
                         0.04 * cos(73* 2 * PI * 1 * t) + 0.09 * sin(73* 2 * PI * 1 * t) + 
                         0.03 * cos(74* 2 * PI * 1 * t) + 0.01 * sin(74* 2 * PI * 1 * t) + 
                         0.76 * cos(75* 2 * PI * 1 * t) + -0.16 * sin(75* 2 * PI * 1 * t) + 
                         0.10 * cos(76* 2 * PI * 1 * t) + -0.01 * sin(76* 2 * PI * 1 * t) + 
                         -0.01 * cos(77* 2 * PI * 1 * t) + 0.00 * sin(77* 2 * PI * 1 * t) + 
                         0.01 * cos(78* 2 * PI * 1 * t)+ 0.03 * sin(78* 2 * PI * 1 * t) + 
                         0.01 * cos(79* 2 * PI * 1 * t) + 0.02 * sin(79* 2 * PI * 1 * t) + 
                         0.02 * cos(80* 2 * PI * 1 * t) + -0.06 * sin(80* 2 * PI * 1 * t) + 
                         -0.05 * cos(81* 2 * PI * 1 * t) + -0.75 * sin(81* 2 * PI * 1 * t) + 
                         -0.01 * cos(82* 2 * PI * 1 * t) + -0.03 * sin(82* 2 * PI * 1 * t) + 
                         0.03 * cos(83* 2 * PI * 1 * t) + -0.03 * sin(83* 2 * PI * 1 * t) + 
                         0.00 * cos(84* 2 * PI * 1 * t) + 0.00 * sin(84* 2 * PI * 1 * t) + 
                         -0.01 * cos(85* 2 * PI * 1 * t) + -0.03 * sin(85* 2 * PI * 1 * t) + 
                         0.02 * cos(86* 2 * PI * 1 * t) + 0.00 * sin(86* 2 * PI * 1 * t) + 
                         -0.02 * cos(87* 2 * PI * 1 * t) + -0.51 * sin(87* 2 * PI * 1 * t) + 
                         -0.73 * cos(88* 2 * PI * 1 * t) + 0.11 * sin(88* 2 * PI * 1 * t) + 
                         -0.05 * cos(89* 2 * PI * 1 * t) + 0.03 * sin(89* 2 * PI * 1 * t) + 
                         0.01 * cos(90* 2 * PI * 1 * t) + 0.00 * sin(90* 2 * PI * 1 * t) + 
                         0.01 * cos(91* 2 * PI * 1 * t) + 0.01 * sin(91* 2 * PI * 1 * t) + 
                         -0.04 * cos(92* 2 * PI * 1 * t) + -0.03 * sin(92* 2 * PI * 1 * t) + 
                         -0.02 * cos(93* 2 * PI * 1 * t) + -0.67 * sin(93* 2 * PI * 1 * t) + 
                         0.43 * cos(94* 2 * PI * 1 * t) + -0.00 * sin(94* 2 * PI * 1 * t) + 
                         -0.02 * cos(95* 2 * PI * 1 * t) + -0.01 * sin(95* 2 * PI * 1 * t) + 
                         -0.01 * cos(96* 2 * PI * 1 * t) + -0.02 * sin(96* 2 * PI * 1 * t) + 
                         0.01 * cos(97* 2 * PI * 1 * t) + -0.02 * sin(97* 2 * PI * 1 * t) + 
                         0.04 * cos(98* 2 * PI * 1 * t) + -0.08 * sin(98* 2 * PI * 1 * t) + 
                         0.02 * cos(99* 2 * PI * 1 * t) + 0.17 * sin(99* 2 * PI * 1 * t) + 
                         0.19 * cos(100* 2 * PI * 1 * t) + -0.03 * sin(100* 2 * PI * 1 * t) + 
                         0.11 * cos(101* 2 * PI * 1 * t) + -0.01 * sin(101* 2 * PI * 1 * t) + 
                         -0.05 * cos(102* 2 * PI * 1 * t) + -0.03 * sin(102* 2 * PI * 1 * t) + 
                         -0.00 * cos(103* 2 * PI * 1 * t) + -0.02 * sin(103* 2 * PI * 1 * t) + 
                         0.00 * cos(104* 2 * PI * 1 * t) + -0.03 * sin(104* 2 * PI * 1 * t) + 
                         0.01 * cos(105* 2 * PI * 1 * t) + -0.00 * sin(105* 2 * PI * 1 * t) + 
                         -0.02 * cos(106* 2 * PI * 1 * t) + 0.56 * sin(106* 2 * PI * 1 * t) + 
                         0.58 * cos(107* 2 * PI * 1 * t) + -0.07 * sin(107* 2 * PI * 1 * t) + 
                         -0.05 * cos(108* 2 * PI * 1 * t) + -0.07 * sin(108* 2 * PI * 1 * t) + 
                         -0.02 * cos(109* 2 * PI * 1 * t) + 0.00 * sin(109* 2 * PI * 1 * t) + 
                         0.02 * cos(110* 2 * PI * 1 * t) + 0.03 * sin(110* 2 * PI * 1 * t) + 
                         -0.01 * cos(111* 2 * PI * 1 * t) + 0.01 * sin(111* 2 * PI * 1 * t) + 
                         -0.00 * cos(112* 2 * PI * 1 * t) + 0.41 * sin(112* 2 * PI * 1 * t) + 
                         0.36 * cos(113* 2 * PI * 1 * t) + -0.01 * sin(113* 2 * PI * 1 * t) + 
                         -0.15 * cos(114* 2 * PI * 1 * t) + 0.02 * sin(114* 2 * PI * 1 * t) + 
                         0.05 * cos(115* 2 * PI * 1 * t) + -0.02 * sin(115* 2 * PI * 1 * t) + 
                         -0.03 * cos(116* 2 * PI * 1 * t)+ -0.04 * sin(116* 2 * PI * 1 * t) + 
                         -0.02 * cos(117* 2 * PI * 1 * t) + 0.01 * sin(117* 2 * PI * 1 * t) + 
                         -0.05 * cos(118* 2 * PI * 1 * t) + 0.39 * sin(118* 2 * PI * 1 * t) + 
                         0.07 * cos(119* 2 * PI * 1 * t) + -0.11 * sin(119* 2 * PI * 1 * t) + 
                         -0.06 * cos(120* 2 * PI * 1 * t) + 0.02 * sin(120* 2 * PI * 1 * t) + 
                         0.02 * cos(121* 2 * PI * 1 * t) + -0.02 * sin(121* 2 * PI * 1 * t) + 
                         -0.00 * cos(122* 2 * PI * 1 * t) + -0.02 * sin(122* 2 * PI * 1 * t) + 
                         0.01 * cos(123* 2 * PI * 1 * t) + -0.05 * sin(123* 2 * PI * 1 * t) + 
                         0.03 * cos(124* 2 * PI * 1 * t) + 0.04 * sin(124* 2 * PI * 1 * t) + 
                         -0.07 * cos(125* 2 * PI * 1 * t) + 0.22 * sin(125* 2 * PI * 1 * t) + 
                         -0.65 * cos(126* 2 * PI * 1 * t) + 0.05 * sin(126* 2 * PI * 1 * t) + 
                         0.04 * cos(127* 2 * PI * 1 * t) + 0.03 * sin(127* 2 * PI * 1 * t) + 
                         -0.01 * cos(128* 2 * PI * 1 * t) + 0.04 * sin(128* 2 * PI * 1 * t) + 
                         0.02 * cos(129* 2 * PI * 1 * t) + 0.05 * sin(129* 2 * PI * 1 * t) + 
                         0.02 * cos(130* 2 * PI * 1 * t) + 0.00 * sin(130* 2 * PI * 1 * t) + 
                         -0.14 * cos(131* 2 * PI * 1 * t) + -1.64 * sin(131* 2 * PI * 1 * t) + 
                         -0.85 * cos(132* 2 * PI * 1 * t) + -0.11 * sin(132* 2 * PI * 1 * t) + 
                         -0.06 * cos(133* 2 * PI * 1 * t) + -0.03 * sin(133* 2 * PI * 1 * t) + 
                         -0.01 * cos(134* 2 * PI * 1 * t) + 0.02 * sin(134* 2 * PI * 1 * t) + 
                         0.02 * cos(135* 2 * PI * 1 * t) + -0.00 * sin(135* 2 * PI * 1 * t)) / 32767 * 255);
    Serial.print(samples_2[i]);
    Serial.print(", ");
  }
}

void generateSample3() {
  // Fill samples_4 with 0
  for (int i = 0; i < numSamples; i++) {
    samples_3[i] = 0;
  }
}

void setup() {
  HMD.begin();
  Serial.begin(115200);
  HMD.Mode(0x03); // PWM INPUT 
  HMD.MotorSelect(0x0A);
  HMD.Library(7); // Change to 6 for LRA motors

  for (int i = 0; i < numPins; i++) {
    pinMode(analogOutPins[i], OUTPUT);
  }
  pinMode(buttonPin, INPUT_PULLUP);
  generateSample0();
  generateSample1();
  generateSample2();
  generateSample3();
  //analogWrite(analogOutPin5, 255);
  //analogWrite(analogOutPin5, 0);  // Start with the output off
}

void loop() {
  int buttonState = digitalRead(buttonPin);

  if (buttonState != oldButtonState && buttonState == HIGH) {
    // Button is pressed, switch the analog value
    // Button is pressed, switch to the next sample
    currentSampleIndex = (currentSampleIndex + 1) % 6; // Cycle through 0, 1, 2

    // Output the appropriate sample based on the current index
    if (currentSampleIndex == 0) {
      activeSample = samples_0;
      Serial.println("Switched to sample 0");
      for (int i = 0; i < numSamples; i++) {
        for (int pin = 0; pin < numPins; pin++) {
          analogWrite(analogOutPins[pin], activeSample[i]);
        }
      }
    } else if (currentSampleIndex == 1) {
      activeSample = samples_3;
      Serial.println("Switched to sample 3");
      for (int i = 0; i < numSamples; i++) {
        for (int pin = 0; pin < numPins; pin++) {
          analogWrite(analogOutPins[pin], activeSample[i]);
        }
      }
    } else if (currentSampleIndex == 2) {
      activeSample = samples_1; 
      Serial.println("Switched to sample 1");
      for (int i = 0; i < numSamples; i++) {
        for (int pin = 0; pin < numPins; pin++) {
          analogWrite(analogOutPins[pin], activeSample[i]);
        }
      }
    } else if (currentSampleIndex == 3) {
      activeSample = samples_3; 
      Serial.println("Switched to sample 3");
      for (int i = 0; i < numSamples; i++) {
        for (int pin = 0; pin < numPins; pin++) {
          analogWrite(analogOutPins[pin], activeSample[i]);
        }
      }
    } else if (currentSampleIndex == 4) {
      activeSample = samples_2; 
      Serial.println("Switched to sample 2");
      for (int i = 0; i < numSamples; i++) {
        for (int pin = 0; pin < numPins; pin++) {
          analogWrite(analogOutPins[pin], activeSample[i]);
        }
      }
    } else if (currentSampleIndex == 5) {
      activeSample = samples_3; 
      Serial.println("Switched to sample 3");
      for (int i = 0; i < numSamples; i++) {
        for (int pin = 0; pin < numPins; pin++) {
          analogWrite(analogOutPins[pin], activeSample[i]);
        }
      }
    } else {
      activeSample = samples_0; 
      Serial.println("Switched to sample 0");
      for (int i = 0; i < numSamples; i++) {
        for (int pin = 0; pin < numPins; pin++) {
          analogWrite(analogOutPins[pin], activeSample[i]);
        }
      }
    }
  }

  oldButtonState = buttonState;
  delay(20);
}
