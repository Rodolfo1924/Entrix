#include <ESP32Servo.h>

const int BUTTON_PIN = 4;
const int LED_VERDE = 2;
const int LED_ROJO = 15;
const int SERVO_PIN = 18;

Servo servo;
unsigned long lastPress = 0;
const unsigned long debounce = 300;

// Variables para temporizador de servo
unsigned long servoOpenTime = 0;
const unsigned long servoDelay = 5000; // 5 segundos abierto

void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LED_VERDE, OUTPUT);
  pinMode(LED_ROJO, OUTPUT);

  servo.attach(SERVO_PIN);
  servo.write(0); // Torniquete cerrado

  apagarLED();
  Serial.println("ESP32 listo. Esperando botón...");
}

void loop() {
  verificarBoton();
  verificarSerial();
  verificarServoTimer(); // 👈 nuevo
}

void verificarBoton() {
  int readButton = digitalRead(BUTTON_PIN);
  if (readButton == LOW) {
    unsigned long now = millis();
    if (now - lastPress > debounce) {
      Serial.println("START");
      lastPress = now;
    }
  }
}

void verificarSerial() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "VALID") {
      encenderVerde();
    } else if (cmd == "INVALID") {
      encenderRojo();
    }
  }
}

void encenderVerde() {
  digitalWrite(LED_VERDE, HIGH);
  digitalWrite(LED_ROJO, LOW);
  servo.write(100); // Torniquete abierto
  servoOpenTime = millis(); // 👈 guarda el tiempo de apertura
}

void encenderRojo() {
  digitalWrite(LED_VERDE, LOW);
  digitalWrite(LED_ROJO, HIGH);
  servo.write(0); // Torniquete cerrado
  servoOpenTime = 0; // 👈 cancela temporizador
}

void verificarServoTimer() {
  if (servoOpenTime > 0 && millis() - servoOpenTime >= servoDelay) {
    // Ya pasaron 5 segundos → cerrar
    servo.write(0);
    apagarLED();
    servoOpenTime = 0;
    Serial.println("Servo regresó automáticamente a cerrado.");
  }
}

void apagarLED() {
  digitalWrite(LED_VERDE, LOW);
  digitalWrite(LED_ROJO, LOW);
}
