/*
 * Arduino Sensor Reader & 16x2 LCD Display Controller
 * Fruit & Food Freshness Analysis System
 *
 * Capabilities:
 * 1. Environmental Sensor Readings:
 *    - Temperature & Humidity: DHT11 / DHT22 (Digital Pin 7)
 *    - Gas / VOC / Air Quality: MQ-135 (Analog Pin A0)
 *    - Telemetry output formatted as JSON Lines: {"temperature_c": 24.5, "humidity_percent": 65.0, "gas_value": 312}
 * 2. Physical Button Trigger on PIN 2 (press button when food is placed at sensor) -> Sends {"button":"pressed","event":"trigger"}
 * 3. 16x2 Character LCD Display Controller:
 *    - Standby / Idle: "Put food near" / "Upload image"
 *    - Incoming Serial Commands: e.g. "LCD:Status: FRESH|Check report\n" or "LCD|Analyzing...|Please wait...\n"
 *    - Strict 16-character line truncation & clean display updates
 */

// =============================================================================
// LCD CONFIGURATION (16x2 Character Display)
// =============================================================================
// Set USE_I2C_LCD to 1 for I2C 16x2 LCD (Address 0x27 or 0x3F via SDA/SCL pins A4/A5)
// Set USE_I2C_LCD to 0 for Standard Parallel 16x2 LCD (RS=12, E=11, D4=5, D5=4, D6=3, D7=6)
#define USE_I2C_LCD 1

#if USE_I2C_LCD
  #include <Wire.h>
  #include <LiquidCrystal_I2C.h>
  // Standard I2C backpack address is 0x27 (or 0x3F)
  LiquidCrystal_I2C lcd(0x27, 16, 2);
#else
  #include <LiquidCrystal.h>
  LiquidCrystal lcd(12, 11, 5, 4, 3, 6);
#endif

// =============================================================================
// TEMPERATURE & HUMIDITY SENSOR CONFIGURATION (DHT11 / DHT22)
// =============================================================================
#include <DHT.h>

#define DHTPIN 7          // Digital pin connected to the DHT sensor data pin
#define DHTTYPE DHT11     // Change to DHT22 if using a DHT22 / AM2302 sensor

DHT dht(DHTPIN, DHTTYPE);

// =============================================================================
// HARDWARE PIN DEFINITIONS
// =============================================================================
const int GAS_SENSOR_PIN = A0;            // Analog input for MQ-135 / Gas sensor
const int TRIGGER_BUTTON_PIN = 2;         // Digital input for push-button (Active LOW with internal pull-up)
const unsigned long READ_INTERVAL = 1000; // Sensor broadcast interval in ms

unsigned long lastReadTime = 0;
int lastButtonState = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long DEBOUNCE_DELAY = 50;

String inputBuffer = "";

// Track last displayed lines to avoid unnecessary LCD redraws / flicker
String currentLcdLine1 = "";
String currentLcdLine2 = "";

// =============================================================================
// LCD HELPER FUNCTION
// =============================================================================
void setLCD(String line1, String line2) {
  // Truncate and sanitize strictly to at most 16 characters
  line1.trim();
  line2.trim();
  if (line1.length() > 16) line1 = line1.substring(0, 16);
  if (line2.length() > 16) line2 = line2.substring(0, 16);

  // Pad lines with spaces to 16 characters to overwrite previous text cleanly
  while (line1.length() < 16) line1 += " ";
  while (line2.length() < 16) line2 += " ";

  if (line1 == currentLcdLine1 && line2 == currentLcdLine2) {
    return; // No change, avoid flicker
  }

  currentLcdLine1 = line1;
  currentLcdLine2 = line2;

  lcd.setCursor(0, 0);
  lcd.print(line1);
  lcd.setCursor(0, 1);
  lcd.print(line2);
}

// =============================================================================
// SETUP
// =============================================================================
void setup() {
  Serial.begin(9600);
  while (!Serial) {
    ; // Wait for serial port to connect
  }

  pinMode(GAS_SENSOR_PIN, INPUT);
  pinMode(TRIGGER_BUTTON_PIN, INPUT_PULLUP);

  // Initialize Temperature & Humidity sensor
  dht.begin();

  // Initialize LCD display
#if USE_I2C_LCD
  lcd.init();
  lcd.backlight();
#else
  lcd.begin(16, 2);
#endif

  // Initial State 1: IDLE / Standby
  setLCD("Put food near", "Upload image");
}

// =============================================================================
// MAIN LOOP
// =============================================================================
void loop() {
  unsigned long currentTime = millis();

  // ---------------------------------------------------------------------------
  // 1. Check Hardware Button Press (with debounce)
  // ---------------------------------------------------------------------------
  int reading = digitalRead(TRIGGER_BUTTON_PIN);
  if (reading != lastButtonState) {
    lastDebounceTime = currentTime;
  }

  if ((currentTime - lastDebounceTime) > DEBOUNCE_DELAY) {
    if (reading == LOW && lastButtonState == HIGH) {
      // Button was clicked down
      Serial.println("{\"button\":\"pressed\",\"event\":\"trigger\"}");
    }
  }
  lastButtonState = reading;

  // ---------------------------------------------------------------------------
  // 2. Read Incoming Serial Commands from PC / Browser (e.g. LCD:Line1|Line2)
  // ---------------------------------------------------------------------------
  while (Serial.available() > 0) {
    char inChar = (char)Serial.read();
    if (inChar == '\n' || inChar == '\r') {
      if (inputBuffer.length() > 0) {
        handleIncomingCommand(inputBuffer);
        inputBuffer = "";
      }
    } else {
      inputBuffer += inChar;
    }
  }

  // ---------------------------------------------------------------------------
  // 3. Periodic Sensor Telemetry Broadcast (JSON Lines)
  // ---------------------------------------------------------------------------
  if (currentTime - lastReadTime >= READ_INTERVAL) {
    lastReadTime = currentTime;

    // Read real sensor telemetry from connected hardware
    float temperature_c = dht.readTemperature();
    float humidity_percent = dht.readHumidity();
    int raw_gas = analogRead(GAS_SENSOR_PIN);

    // Emit standard JSON Lines packet (exact format expected by python/browser)
    Serial.print("{\"temperature_c\":");
    if (isnan(temperature_c)) {
      Serial.print("null");
    } else {
      Serial.print(temperature_c, 1);
    }

    Serial.print(",\"humidity_percent\":");
    if (isnan(humidity_percent)) {
      Serial.print("null");
    } else {
      Serial.print(humidity_percent, 1);
    }

    Serial.print(",\"gas_value\":");
    Serial.print(raw_gas);
    Serial.println("}");
  }
}

// =============================================================================
// COMMAND HANDLER
// =============================================================================
void handleIncomingCommand(String cmd) {
  cmd.trim();
  // Support both "LCD:Line1|Line2" and "LCD|Line1|Line2"
  if (cmd.startsWith("LCD:") || cmd.startsWith("LCD|")) {
    String content = cmd.substring(4);
    int splitIdx = content.indexOf('|');
    String line1 = splitIdx >= 0 ? content.substring(0, splitIdx) : content;
    String line2 = splitIdx >= 0 ? content.substring(splitIdx + 1) : "";

    setLCD(line1, line2);
  }
}
