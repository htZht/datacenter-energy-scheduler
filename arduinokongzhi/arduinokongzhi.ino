// arduino_sensors.ino
#include <Wire.h>
#include <BH1750.h>

BH1750 lightMeter;

const int windPin = 2; // 风速传感器接 D2（中断引脚）
volatile unsigned long pulseCount = 0;
float windSpeed = 0.0;

void countPulse() {
  pulseCount++;
}

void setup() {
  Serial.begin(9600);
  Wire.begin();
  lightMeter.begin();
  
  pinMode(windPin, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(windPin), countPulse, FALLING);
}

void loop() {
  // 读取光照 (lux)
  float lux = lightMeter.readLightLevel();
  
  // 计算风速：假设每秒脉冲数 × 0.8 = m/s（根据传感器校准）
  noInterrupts();
  unsigned long count = pulseCount;
  pulseCount = 0;
  interrupts();
  
  windSpeed = count * 0.8; // 示例转换系数，请按实际校准
  
  // 输出格式: "LUX, WIND" （如 "12500, 4.2"）
  Serial.print(lux);
  Serial.print(",");
  Serial.println(windSpeed);
  
  delay(1000); // 每秒发送一次
}