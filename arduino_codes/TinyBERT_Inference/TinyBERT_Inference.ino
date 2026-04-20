/*
  TinyBERT_Inference.ino - PROGMEM version
  
  Weights stored in FLASH (32KB) not SRAM (2KB)
  using PROGMEM keyword — this fixes the memory error.
*/

#include <avr/pgmspace.h>
#include "tinybert_weights.h"

#define BAUD_RATE       9600
#define EMBEDDING_SIZE  312
#define BYTES_PER_FLOAT 4
#define TOTAL_BYTES     (EMBEDDING_SIZE * BYTES_PER_FLOAT)

float embedding[EMBEDDING_SIZE];

int classify(float* emb) {
  float scores[NUM_LABELS];
  for (int label = 0; label < NUM_LABELS; label++) {
    float score = pgm_read_float(&classifier_bias[label]);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
      float w = pgm_read_float(&classifier_weights[label][i]);
      score += w * emb[i];
    }
    scores[label] = score;
  }
  return (scores[1] > scores[0]) ? 1 : 0;
}

void setup() {
  Serial.begin(BAUD_RATE);
  pinMode(LED_BUILTIN, OUTPUT);
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_BUILTIN, HIGH); delay(200);
    digitalWrite(LED_BUILTIN, LOW);  delay(200);
  }
  Serial.println("READY");
}

void loop() {
  if (Serial.available() >= TOTAL_BYTES) {
    byte* buf = (byte*)embedding;
    for (int i = 0; i < TOTAL_BYTES; i++) {
      buf[i] = Serial.read();
    }
    int prediction = classify(embedding);
    if (prediction == 1) {
      Serial.println("POSITIVE");
      digitalWrite(LED_BUILTIN, HIGH);
    } else {
      Serial.println("NEGATIVE");
      digitalWrite(LED_BUILTIN, LOW);
    }
    delay(200);
    digitalWrite(LED_BUILTIN, LOW);
  }
}