#include <PDM.h>
#include <Arduino_APDS9960.h>

// PDM audio buffers
static const char channels = 1;
static const int frequency = 16000;
short sampleBuffer[16000];
volatile int samplesRead;
short transmitBuffer[4096];
int transmitBufferIndex = 0;

void setup() {
  Serial.begin(115200);
  Serial.println("PDM microphone streaming started.");

  // Initialize gesture sensor
  if (!APDS.begin()) {  // <-- static begin() call, no object needed
    Serial.println("Failed to initialize APDS-9960!");
    while (1);
  }
  Serial.println("APDS-9960 initialized.");

  // Initialize PDM microphone
  PDM.onReceive(onPDMdata);
  if (!PDM.begin(channels, frequency)) {
    Serial.println("Failed to start PDM!");
    while (1);
  }
  PDM.setGain(80);
}

void loop() {
  // Gesture detection
  if (APDS.gestureAvailable()) {
    int gesture = APDS.readGesture();
    if (gesture == GESTURE_DOWN) {
      Serial.println("STOP"); // notify Python
      while (1);              // stop streaming
    }
  }

  // Send audio buffer
  if (transmitBufferIndex > 0) {
    Serial.write((byte*)transmitBuffer, transmitBufferIndex * sizeof(short));
    transmitBufferIndex = 0;
  }
}

// PDM callback
void onPDMdata() {
  int bytesAvailable = PDM.available();
  PDM.read(sampleBuffer, bytesAvailable);
  samplesRead = bytesAvailable / 2;

  for (int i = 0; i < samplesRead; i++) {
    transmitBuffer[transmitBufferIndex++] = sampleBuffer[i];
    if (transmitBufferIndex == sizeof(transmitBuffer)/sizeof(short)) {
      Serial.write((byte*)transmitBuffer, sizeof(transmitBuffer));
      transmitBufferIndex = 0;
    }
  }
}
