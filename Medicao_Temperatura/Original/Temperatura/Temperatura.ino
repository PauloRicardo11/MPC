#include <Wire.h>
#include <Adafruit_SHT31.h>

Adafruit_SHT31 interior = Adafruit_SHT31();
Adafruit_SHT31 exterior = Adafruit_SHT31();

void setup() {
  Serial.begin(9600);

  if (!interior.begin(0x44)) {
    Serial.println("NAN");
    while (1);
  }

  if (!exterior.begin(0x45)) {
    Serial.println("NAN");
    while (1);
  }

  Serial.println("OK");
}

void loop() {
  float temp_int = interior.readTemperature();
  float hum_int = interior.readHumidity();

  float temp_ext = exterior.readTemperature();
  float hum_ext = exterior.readHumidity();

  if (isnan(temp_int) || isnan(hum_int) ||
      isnan(temp_ext) || isnan(hum_ext)) {
    Serial.println("NAN");
  } else {
    Serial.print(temp_int);
    Serial.print(",");
    Serial.print(hum_int);
    Serial.print(",");
    Serial.print(temp_ext);
    Serial.print(",");
    Serial.println(hum_ext);
  }

  delay(60000);
}