// emg_imu_stream.ino  -  EMG Mouse firmware v2 (serial protocol v2, see ../PROTOCOL.md)
//
// Board: "ESP32 Dev Module", core "esp32 by Espressif" 3.x.  Only the standard Wire library.
//
//  * EMG: GPIO32 (ch1) and GPIO33 (ch2) read back-to-back at exactly 1000 Hz.
//         A hardware timer ISR wakes a top-priority sampler task pinned to core 1.
//         The sample counter counts timer ticks, so it IS the timebase: a sample that
//         cannot be sent is dropped but the counter still advances (the host sees a gap).
//  * IMU: one MPU-6050 at 0x68 on core 0, read every 5th EMG tick (=200 Hz, phase-locked
//         to the EMG clock). Each IMU sample carries the EMG counter at read time.
//         A missing/failed IMU never stops EMG: it is reported in STATUS and retried every 1 s.
//  * Serial: 921600 baud, framed packets with CRC-8, written by loop() (lowest priority).
//
// SAFETY: while electrodes are on a person, power the ESP32 from a battery, a power bank
// or an unplugged laptop. Never from a wall-powered PC.

#include <Arduino.h>
#include <Wire.h>
#include <atomic>

#if !defined(ESP_ARDUINO_VERSION_MAJOR) || ESP_ARDUINO_VERSION_MAJOR < 3
#error "This sketch needs the ESP32 Arduino core 3.x (Boards Manager: esp32 by Espressif)"
#endif

#define FW_VERSION "2.0.0"

// ---------------------------------------------------------------- configuration
static const uint8_t  PROTO_VERSION = 2;
static const int      PIN_CH1 = 32;
static const int      PIN_CH2 = 33;
static const int      PIN_SDA = 21;
static const int      PIN_SCL = 22;
static const uint8_t  MPU_ADDR = 0x68;
static const uint32_t BAUD = 921600;
static const uint16_t FS_EMG = 1000;
static const uint16_t FS_IMU = 200;
static const uint32_t IMU_DIV = FS_EMG / FS_IMU;   // 5
static const uint8_t  EMG_BATCH = 5;               // EMG samples per packet (5 ms)
static const uint8_t  ACCEL_FS_G = 4;
static const uint16_t GYRO_FS_DPS = 500;
static const uint8_t  DLPF_CFG = 3;                // accel 44 Hz / gyro 42 Hz

// packet types
static const uint8_t T_EMG = 0x01, T_IMU = 0x02, T_STATUS = 0x03, T_INFO = 0x04;

// IMU states
enum : uint8_t { IMU_OK = 0, IMU_MISSING = 1, IMU_FAILED = 2, IMU_INIT = 3 };

// ---------------------------------------------------------------- shared state
static const uint32_t EMG_RING = 1024;             // power of two, ~1 s of samples
static const uint32_t IMU_RING = 64;
struct EmgPair { uint16_t a, b; };
struct ImuSample { uint32_t tag; uint8_t raw[14]; };

static EmgPair   emgRing[EMG_RING];
static ImuSample imuRing[IMU_RING];
static std::atomic<uint32_t> emgHead{0};           // == EMG sample counter (samples taken)
static std::atomic<uint32_t> imuHead{0};

static TaskHandle_t samplerTask = nullptr;
static TaskHandle_t imuTask = nullptr;
static hw_timer_t*  timer = nullptr;

// diagnostics (written by one task each, read by loop)
static std::atomic<uint32_t> emgLateTicks{0};    // ticks the sampler handled late (bunched)
static std::atomic<uint32_t> emgMaxJitterUs{0};  // max |interval-1000us| since last STATUS
static std::atomic<uint32_t> emgOverflow{0};     // samples dropped because the writer fell behind
static std::atomic<uint32_t> imuOverflow{0};
static std::atomic<uint8_t>  imuState{IMU_INIT};
static std::atomic<uint8_t>  imuWhoAmI{0};
static std::atomic<uint16_t> i2cKhz{400};
static std::atomic<uint32_t> imuReads{0}, imuErrors{0}, imuRecoveries{0};

// ---------------------------------------------------------------- CRC-8 (poly 0x07, init 0)
static uint8_t crcTable[256];
static void crcInit() {
  for (int i = 0; i < 256; i++) {
    uint8_t c = i;
    for (int j = 0; j < 8; j++) c = (c & 0x80) ? (uint8_t)((c << 1) ^ 0x07) : (uint8_t)(c << 1);
    crcTable[i] = c;
  }
}

// ---------------------------------------------------------------- EMG sampling (core 1)
void ARDUINO_ISR_ATTR onTimer() {
  BaseType_t hp = pdFALSE;
  vTaskNotifyGiveFromISR(samplerTask, &hp);
  if (hp) portYIELD_FROM_ISR();
}

static void samplerLoop(void*) {
  // attach the timer from this task so its ISR runs on core 1 too
  timer = timerBegin(1000000);                 // 1 MHz tick
  timerAttachInterrupt(timer, &onTimer);
  timerAlarm(timer, 1000000 / FS_EMG, true, 0);
  uint32_t c = 0;
  int64_t lastUs = esp_timer_get_time();
  for (;;) {
    uint32_t n = ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    if (n > 1) emgLateTicks += n - 1;
    for (uint32_t k = 0; k < n; k++) {
      int64_t now = esp_timer_get_time();
      uint16_t a = analogRead(PIN_CH1);
      uint16_t b = analogRead(PIN_CH2);
      if (n == 1) {
        uint32_t d = (uint32_t)llabs((now - lastUs) - 1000);
        if (d > emgMaxJitterUs) emgMaxJitterUs = d;
      }
      lastUs = now;
      emgRing[c & (EMG_RING - 1)] = {a, b};
      c++;
      emgHead.store(c, std::memory_order_release);
      if (c % IMU_DIV == 0 && imuTask) xTaskNotifyGive(imuTask);
    }
  }
}

// ---------------------------------------------------------------- MPU-6050 (core 0)
static bool i2cWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static bool i2cRead(uint8_t reg, uint8_t* buf, uint8_t len) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU_ADDR, (size_t)len) != len) return false;
  for (uint8_t i = 0; i < len; i++) buf[i] = Wire.read();
  return true;
}

// Free a slave that is holding SDA low (e.g. after a glitch mid-transfer).
static void i2cBusRecover() {
  Wire.end();
  pinMode(PIN_SDA, INPUT_PULLUP);
  pinMode(PIN_SCL, OUTPUT_OPEN_DRAIN);
  for (int i = 0; i < 9 && digitalRead(PIN_SDA) == LOW; i++) {
    digitalWrite(PIN_SCL, LOW);  delayMicroseconds(5);
    digitalWrite(PIN_SCL, HIGH); delayMicroseconds(5);
  }
  // STOP: SDA low->high while SCL high
  pinMode(PIN_SDA, OUTPUT_OPEN_DRAIN);
  digitalWrite(PIN_SDA, LOW);  delayMicroseconds(5);
  digitalWrite(PIN_SCL, HIGH); delayMicroseconds(5);
  digitalWrite(PIN_SDA, HIGH); delayMicroseconds(5);
}

static bool mpuInit() {
  Wire.begin(PIN_SDA, PIN_SCL, (uint32_t)i2cKhz * 1000);
  Wire.setTimeOut(10);
  uint8_t who = 0;
  if (!i2cRead(0x75, &who, 1)) { imuWhoAmI = 0; return false; }
  imuWhoAmI = who;                                   // 0x68 genuine; clones report 0x70/0x72/0x98
  if (!i2cWrite(0x6B, 0x80)) return false;           // PWR_MGMT_1: device reset
  vTaskDelay(pdMS_TO_TICKS(100));
  if (!i2cWrite(0x6B, 0x01)) return false;           // wake, clock = PLL with X gyro
  vTaskDelay(pdMS_TO_TICKS(10));
  // 1 kHz internal rate (SMPLRT_DIV=0 with DLPF on); we read it at 200 Hz on our own clock,
  // so the two clocks never beat against each other and the data is at most 1 ms old.
  bool ok = i2cWrite(0x19, 0x00)                     // SMPLRT_DIV
         && i2cWrite(0x1A, DLPF_CFG)                 // CONFIG: DLPF ~44 Hz
         && i2cWrite(0x1B, 0x08)                     // GYRO_CONFIG: +-500 dps
         && i2cWrite(0x1C, 0x08);                    // ACCEL_CONFIG: +-4 g
  if (!ok) return false;
  uint8_t cfg[4];
  if (!i2cRead(0x19, cfg, 4)) return false;
  return cfg[0] == 0x00 && (cfg[1] & 0x07) == DLPF_CFG && (cfg[2] & 0x18) == 0x08 && (cfg[3] & 0x18) == 0x08;
}

static void imuLoop(void*) {
  bool everOk = false;
  uint32_t consecFail = 0;
  uint32_t lastTry = 0;
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    if (imuState != IMU_OK) {
      uint32_t ms = millis();
      if (lastTry != 0 && ms - lastTry < 1000) continue;
      lastTry = ms;
      if (mpuInit()) {
        imuState = IMU_OK;
        consecFail = 0;
        if (everOk) imuRecoveries++;
        everOk = true;
      } else {
        imuState = everOk ? IMU_FAILED : IMU_MISSING;
        i2cBusRecover();
        if (everOk) i2cKhz = 100;                    // flaky long leads: retry slower
      }
      continue;
    }
    ImuSample s;
    s.tag = emgHead.load(std::memory_order_acquire);
    if (i2cRead(0x3B, s.raw, 14)) {
      consecFail = 0;
      imuReads++;
      uint32_t h = imuHead.load(std::memory_order_relaxed);
      imuRing[h & (IMU_RING - 1)] = s;
      imuHead.store(h + 1, std::memory_order_release);
    } else {
      imuErrors++;
      if (++consecFail >= 5) {                       // 25 ms of failures: declare failed, recover
        imuState = IMU_FAILED;
        i2cKhz = 100;
        i2cBusRecover();
        lastTry = millis();
      }
    }
  }
}

// ---------------------------------------------------------------- serial writer (loop, core 1)
static uint8_t pkt[256];

static void sendPacket(uint8_t type, const uint8_t* payload, uint8_t len) {
  pkt[0] = 0xA5; pkt[1] = 0x5A; pkt[2] = type; pkt[3] = len;
  memcpy(pkt + 4, payload, len);
  uint8_t c = 0;
  for (int i = 2; i < 4 + len; i++) c = crcTable[c ^ pkt[i]];
  pkt[4 + len] = c;
  Serial.write(pkt, 5 + len);
}

static inline void put16(uint8_t*& p, uint16_t v) { *p++ = v; *p++ = v >> 8; }
static inline void put32(uint8_t*& p, uint32_t v) { for (int i = 0; i < 4; i++) *p++ = v >> (8 * i); }

static void sendInfo() {
  uint8_t b[64], *p = b;
  *p++ = PROTO_VERSION;
  put16(p, FS_EMG); put16(p, FS_IMU);
  *p++ = EMG_BATCH; *p++ = ACCEL_FS_G; put16(p, GYRO_FS_DPS); *p++ = DLPF_CFG;
  const char* v = FW_VERSION;
  size_t n = strlen(v);
  memcpy(p, v, n); p += n;
  sendPacket(T_INFO, b, p - b);
}

static void sendStatus() {
  uint8_t b[64], *p = b;
  put32(p, millis());
  put32(p, emgHead.load(std::memory_order_acquire));
  *p++ = imuState;
  *p++ = imuWhoAmI;
  put16(p, i2cKhz);
  put32(p, imuReads); put32(p, imuErrors); put32(p, imuRecoveries);
  put32(p, emgLateTicks); put32(p, emgOverflow); put32(p, imuOverflow);
  uint32_t j = emgMaxJitterUs.exchange(0);
  put16(p, j > 65535 ? 65535 : j);
  sendPacket(T_STATUS, b, p - b);
}

static uint32_t emgSent = 0, imuSent = 0;
static uint32_t nextStatus = 0, nextInfo = 0;

void setup() {
  crcInit();
  Serial.setTxBufferSize(8192);
  Serial.begin(BAUD);
  analogReadResolution(12);
  analogSetPinAttenuation(PIN_CH1, ADC_11db);
  analogSetPinAttenuation(PIN_CH2, ADC_11db);
  analogRead(PIN_CH1); analogRead(PIN_CH2);         // init the ADC channels before sampling starts
  delay(50);
  sendInfo();
  xTaskCreatePinnedToCore(imuLoop, "imu", 4096, nullptr, 5, &imuTask, 0);
  xTaskCreatePinnedToCore(samplerLoop, "emg", 4096, nullptr, configMAX_PRIORITIES - 1, &samplerTask, 1);
  nextStatus = millis() + 1000;
  nextInfo = millis() + 5000;
}

void loop() {
  // EMG: whole batches only
  uint32_t head = emgHead.load(std::memory_order_acquire);
  if (head - emgSent > EMG_RING - 2 * EMG_BATCH) {    // fell ~1 s behind: skip ahead (host sees a gap)
    uint32_t skip = (head - EMG_BATCH * 20) - emgSent;
    skip -= skip % EMG_BATCH;
    emgOverflow += skip;
    emgSent += skip;
  }
  while (head - emgSent >= EMG_BATCH) {
    uint8_t b[4 + 4 * EMG_BATCH], *p = b;
    put32(p, emgSent);
    for (uint8_t i = 0; i < EMG_BATCH; i++) {
      EmgPair e = emgRing[(emgSent + i) & (EMG_RING - 1)];
      put16(p, e.a); put16(p, e.b);
    }
    sendPacket(T_EMG, b, sizeof b);
    emgSent += EMG_BATCH;
  }
  // IMU
  uint32_t ih = imuHead.load(std::memory_order_acquire);
  if (ih - imuSent > IMU_RING - 4) { imuOverflow += (ih - 4) - imuSent; imuSent = ih - 4; }
  while (imuSent != ih) {
    ImuSample s = imuRing[imuSent & (IMU_RING - 1)];
    uint8_t b[18], *p = b;
    put32(p, s.tag);
    memcpy(p, s.raw, 14);
    sendPacket(T_IMU, b, sizeof b);
    imuSent++;
  }
  // housekeeping
  uint32_t ms = millis();
  if ((int32_t)(ms - nextStatus) >= 0) { sendStatus(); nextStatus += 1000; }
  if ((int32_t)(ms - nextInfo) >= 0) { sendInfo(); nextInfo += 5000; }
  while (Serial.available()) {
    int ch = Serial.read();
    if (ch == 'i') sendInfo();
  }
  vTaskDelay(1);                                      // 1 ms; lets idle task run (watchdog)
}
