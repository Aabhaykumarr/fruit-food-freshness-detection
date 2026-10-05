/*
 * Arduino Sensor Reader & 16x2 LCD Display Controller
 * Fruit & Food Freshness Analysis System
 *
 * Capabilities:
 * 1. Environmental Sensor Readings:
 *    - Temperature & Humidity: DHT11 (DATA pin A0)
 *    - Soil Moisture: Analog on A1, Digital on A2
 *    - Telemetry output formatted as JSON Lines:
 *      {"temperature_c": 24.5, "humidity_percent": 65.0, "gas_value": null,
 *       "soil_moisture_raw": 512, "soil_moisture_dry": 0}
 * 2. Physical Button Trigger on PIN 2 (press button when food is placed at sensor)
 *    -> Sends {"button":"pressed","event":"trigger"}
 * 3. 16x2 Character LCD Display Controller:
 *    - Standby / Idle (before analysis): "Put food near" / "Upload image"
 *    - Analyzing (on Analyze click): "Analyzing..." / "Please wait..."
 *    - Analysis Result (completed): Rotates between Result/Status screen & Temp/Hum screen
 *      (Keeps result visible until next analysis; no soil/gas shown on LCD)
 *    - Strict 16-character line truncation & clean display updates
 *
 * Pin assignments (confirmed hardware):
 *   DHT11 DATA  -> A0
 *   Soil AO     -> A1
 *   Soil DO     -> A2
 *   Button      -> D2 (active LOW, internal pull-up)
 *   LCD I2C SDA -> A4
 *   LCD I2C SCL -> A5
 */

// =============================================================================
// LCD CONFIGURATION (16x2 Character Display)
// =============================================================================
// Set USE_I2C_LCD to 1 for I2C 16x2 LCD (Address 0x27 via SDA/SCL pins A4/A5)
// Set USE_I2C_LCD to 0 for Standard Parallel 16x2 LCD (RS=12, E=11, D4=5, D5=4, D6=3, D7=6)
#define USE_I2C_LCD 1

#if USE_I2C_LCD
  #include <Wire.h>
  #include <LiquidCrystal_I2C.h>
  LiquidCrystal_I2C lcd(0x27, 16, 2);
#else
  #include <LiquidCrystal.h>
  LiquidCrystal lcd(12, 11, 5, 4, 3, 6);
#endif

// =============================================================================
// TEMPERATURE & HUMIDITY SENSOR CONFIGURATION (DHT11 on A0)
// =============================================================================
#include <DHT.h>

#define DHTPIN A0         // DHT11 DATA pin wired to analog header A0
#define DHTTYPE DHT11

DHT dht(DHTPIN, DHTTYPE);

// =============================================================================
// HARDWARE PIN DEFINITIONS
// =============================================================================
const int SOIL_ANALOG_PIN  = A1;  // Soil moisture module analog output
const int SOIL_DIGITAL_PIN = A2;  // Soil moisture module digital output (dry threshold)
const int TRIGGER_BUTTON_PIN = 2; // Digital input for push-button (Active LOW with internal pull-up)

const unsigned long READ_INTERVAL = 2000;  // Sensor broadcast interval in ms

unsigned long lastReadTime = 0;
int lastButtonState = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long DEBOUNCE_DELAY = 50;

String inputBuffer = "";

// Latest sensor values (updated each read cycle)
float lastTemp = NAN;
float lastHum  = NAN;
int   lastSoilRaw = 0;
int   lastSoilDry = 0;

// Track last displayed lines to avoid unnecessary LCD redraws / flicker
String currentLcdLine1 = "";
String currentLcdLine2 = "";

// LCD display state management:
//   1. LCD_STATE_IDLE: Shown before analysis starts (including after image upload).
//      Line 1: "Put food near"
//      Line 2: "Upload image"
//   2. LCD_STATE_ANALYZING: Shown as soon as user presses Analyze.
//      Line 1: "Analyzing..."
//      Line 2: "Please wait..."
//   3. LCD_STATE_RESULT: Shown when analysis completes.
//      Rotates between:
//        Screen 1 (Result/Status): e.g. "Status: FRESH" / "Check report"
//        Screen 2 (Captured Temp/Hum): e.g. "Temp: 28.5 C" / "Hum:  40.7 %"
//      Keeps rotating until a new analysis begins ("Analyzing...").
//      Does not show soil moisture or gas readings on LCD.
enum LcdState { LCD_STATE_IDLE, LCD_STATE_ANALYZING, LCD_STATE_RESULT };
LcdState lcdState = LCD_STATE_IDLE;

String resultLine1 = "";
String resultLine2 = "";

// Captured sensor readings for the active analysis
float capturedTemp = NAN;
float capturedHum  = NAN;

// Result screen rotation timer (alternates every 3 seconds between status and temp/hum)
unsigned long lastResultRotateTime = 0;
const unsigned long RESULT_ROTATE_MS = 3000;
bool showTempHumScreen = false;

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

  pinMode(SOIL_ANALOG_PIN, INPUT);
  pinMode(SOIL_DIGITAL_PIN, INPUT);
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

  // Initial State: IDLE / Standby
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
    lastTemp    = dht.readTemperature();
    lastHum     = dht.readHumidity();
    lastSoilRaw = analogRead(SOIL_ANALOG_PIN);
    lastSoilDry = digitalRead(SOIL_DIGITAL_PIN);

    // Emit standard JSON Lines packet (exact format expected by python/browser)
    // Existing fields: temperature_c, humidity_percent, gas_value
    // New additive fields: soil_moisture_raw, soil_moisture_dry
    Serial.print("{\"temperature_c\":");
    if (isnan(lastTemp)) {
      Serial.print("null");
    } else {
      Serial.print(lastTemp, 1);
    }

    Serial.print(",\"humidity_percent\":");
    if (isnan(lastHum)) {
      Serial.print("null");
    } else {
      Serial.print(lastHum, 1);
    }

    // No gas sensor connected (A0 is DHT11); send null to keep protocol compatible
    Serial.print(",\"gas_value\":null");

    Serial.print(",\"soil_moisture_raw\":");
    Serial.print(lastSoilRaw);

    Serial.print(",\"soil_moisture_dry\":");
    Serial.print(lastSoilDry);

    Serial.println("}");
  }

  // ---------------------------------------------------------------------------
  // 4. LCD Display Management
  // ---------------------------------------------------------------------------
  updateLcdDisplay(currentTime);
}

// =============================================================================
// LCD DISPLAY UPDATE
// =============================================================================
void updateLcdDisplay(unsigned long currentTime) {
  if (lcdState == LCD_STATE_IDLE) {
    // 1. Before analysis starts: strictly show "Put food near" / "Upload image"
    setLCD("Put food near", "Upload image");
  } else if (lcdState == LCD_STATE_ANALYZING) {
    // 2. While analyzing: strictly show "Analyzing..." / "Please wait..."
    setLCD("Analyzing...", "Please wait...");
  } else if (lcdState == LCD_STATE_RESULT) {
    // 3. When analysis completes: rotate between Result/Status screen and Temp/Hum screen
    // Keep result and readings visible until another analysis begins
    if (currentTime - lastResultRotateTime >= RESULT_ROTATE_MS) {
      lastResultRotateTime = currentTime;
      showTempHumScreen = !showTempHumScreen;
    }

    if (showTempHumScreen) {
      // Screen 2: Temperature & Humidity (guaranteed <= 16 chars per line)
      String tStr;
      if (isnan(capturedTemp)) {
        tStr = "Temp: --.- C";
      } else {
        tStr = "Temp: " + String(capturedTemp, 1) + " C";
      }

      String hStr;
      if (isnan(capturedHum)) {
        hStr = "Hum:  --.- %";
      } else {
        hStr = "Hum:  " + String(capturedHum, 1) + " %";
      }

      setLCD(tStr, hStr);
    } else {
      // Screen 1: Result / Status screen
      setLCD(resultLine1, resultLine2);
    }
  }
}

// =============================================================================
// COMMAND HANDLER
// =============================================================================
void handleIncomingCommand(String cmd) {
  cmd.trim();
  int lcdIdx = cmd.indexOf("LCD:");
  if (lcdIdx < 0) {
    lcdIdx = cmd.indexOf("LCD|");
  }
  if (lcdIdx >= 0) {
    String content = cmd.substring(lcdIdx + 4);
    int splitIdx = content.indexOf('|');
    String line1 = splitIdx >= 0 ? content.substring(0, splitIdx) : content;
    String line2 = splitIdx >= 0 ? content.substring(splitIdx + 1) : "";
    line1.trim();
    line2.trim();

    if (line1.equalsIgnoreCase("Put food near") || line1.equalsIgnoreCase("idle") ||
        line1.equalsIgnoreCase("Ready to scan") || line1.startsWith("Arduino") ||
        line1.startsWith("Connect") || line1.startsWith("Temp:")) {
      lcdState = LCD_STATE_IDLE;
      setLCD("Put food near", "Upload image");
    } else if (line1.startsWith("Analyzing")) {
      lcdState = LCD_STATE_ANALYZING;
      // Capture the current temperature and humidity readings at the time of analysis
      capturedTemp = lastTemp;
      capturedHum  = lastHum;
      setLCD("Analyzing...", "Please wait...");
    } else {
      // Analysis result status (e.g. "Status: FRESH", "Status: SPOILED", "Not food/fruit", etc.)
      lcdState = LCD_STATE_RESULT;
      resultLine1 = line1;
      resultLine2 = line2;
      // If captured sensor readings were not set during analyzing, latch them now
      if (isnan(capturedTemp) && !isnan(lastTemp)) {
        capturedTemp = lastTemp;
        capturedHum  = lastHum;
      }
      lastResultRotateTime = millis();
      showTempHumScreen = false;
      setLCD(resultLine1, resultLine2);
    }
  }
}
