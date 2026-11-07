// emg_stream.ino
// Streams both EMG channels to the laptop at exactly 1000 samples/s.
// Channel 1 = D32 (GPIO32), Channel 2 = D33 (GPIO33).
//
// Both channels are read back-to-back in the same tick (~20 µs apart), so every
// packet holds one time-aligned pair. Each packet also carries a counter, so the
// laptop can tell if anything was lost.
//
// Packet (9 bytes): 0xAA 0x55 | counter (u16) | ch1 (u16) | ch2 (u16) | xor checksum
// Baud: 921600  (if your board's USB chip struggles, use 460800 here AND in Python --baud)

const int PIN_CH1 = 32;
const int PIN_CH2 = 33;
const uint32_t PERIOD_US = 1000;   // 1 kHz

uint16_t counter = 0;
uint32_t nextT;

void setup() {
  Serial.begin(921600);
  analogReadResolution(12);          // 0..4095
  analogSetAttenuation(ADC_11db);    // full 0..3.3 V range
  nextT = micros();
}

void loop() {
  uint32_t now = micros();
  if ((int32_t)(now - nextT) < 0) return;          // not time yet
  if ((int32_t)(now - nextT) > 50000) nextT = now; // fell way behind: resync
  nextT += PERIOD_US;

  uint16_t a = analogRead(PIN_CH1);
  uint16_t b = analogRead(PIN_CH2);

  uint8_t p[9];
  p[0] = 0xAA; p[1] = 0x55;
  p[2] = counter & 0xFF; p[3] = counter >> 8;
  p[4] = a & 0xFF;       p[5] = a >> 8;
  p[6] = b & 0xFF;       p[7] = b >> 8;
  p[8] = p[2] ^ p[3] ^ p[4] ^ p[5] ^ p[6] ^ p[7];
  Serial.write(p, 9);
  counter++;
}
