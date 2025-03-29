#include <SPI.h>
#include <SD.h>
#include "genann.h"

#define INPUTS 9
#define HIDDEN 9
#define OUTPUTS 7
#define MAX_EPOCHS 10
#define SSL_EPOCHS 10
#define TRAIN_RATIO 0.8

#if defined(ESP32)
  #define CS_PIN 5
#elif defined(ESP8266)
  #define CS_PIN 4
#else
  #define CS_PIN 10  // fallback for other boards
#endif

genann *ann;

// Count data rows (excluding header)
int countLines(const char *path) {
  File file = SD.open(path);
  if (!file) return 0;
  int count = 0;
  while (file.available()) {
    file.readStringUntil('\n');
    count++;
  }
  file.close();
  return count - 1;
}

// Evaluate accuracy/loss (for AE or classifier)
void evaluateFile(const char *path, int startLine, int endLine, bool hasLabel, float &acc, float &loss) {
  File file = SD.open(path);
  if (!file) { acc = 0; loss = 0; return; }

  file.readStringUntil('\n'); // skip header

  int correct = 0;
  float total_loss = 0;
  int line_num = 0;
  int evaluated = 0;

  while (file.available()) {
    String line = file.readStringUntil('\n');
    if (line_num >= startLine && line_num < endLine) {
      char buf[200];
      line.toCharArray(buf, 200);
      char *token = strtok(buf, ",");

      genann_type inputs[INPUTS];
      for (int i = 0; i < INPUTS; i++) {
        inputs[i] = atof(token);
        token = strtok(NULL, ",");
      }

      if (hasLabel) {
        int label = atoi(token);
        const genann_type *output = genann_run(ann, inputs);

        int pred = 0;
        genann_type max_val = output[0];
        for (int j = 1; j < OUTPUTS; j++) {
          if (output[j] > max_val) {
            max_val = output[j];
            pred = j;
          }
        }

        if (pred == label) correct++;

        for (int j = 0; j < OUTPUTS; j++) {
          genann_type target = (j == label) ? 1.0 : 0.0;
          genann_type diff = output[j] - target;
          total_loss += diff * diff;
        }
      } else {
        const genann_type *output = genann_run(ann, inputs);
        for (int j = 0; j < INPUTS; j++) {
          genann_type diff = output[j] - inputs[j];
          total_loss += diff * diff;
        }
      }

      evaluated++;
    }
    line_num++;
  }

  file.close();
  acc = hasLabel ? (100.0 * correct / evaluated) : 0;
  loss = total_loss / evaluated;
}

// Pretrain Autoencoder
void trainAutoencoder(const char *path) {
  int total = countLines(path);
  int split = total * TRAIN_RATIO;

  for (int epoch = 0; epoch < SSL_EPOCHS; epoch++) {
    File file = SD.open(path);
    if (!file) return;

    file.readStringUntil('\n'); // skip header
    int line_num = 0;
    while (file.available() && line_num < split) {
      String line = file.readStringUntil('\n');
      char buf[200];
      line.toCharArray(buf, 200);
      char *token = strtok(buf, ",");

      genann_type inputs[INPUTS];
      for (int i = 0; i < INPUTS; i++) {
        inputs[i] = atof(token);
        token = strtok(NULL, ",");
      }

      genann_train(ann, inputs, inputs, 0.1);
      line_num++;
    }
    file.close();

    float train_loss, val_loss, acc;
    evaluateFile(path, 0, split, false, acc, train_loss);
    evaluateFile(path, split, total, false, acc, val_loss);

    Serial.print("SSL Epoch "); Serial.print(epoch + 1);
    Serial.print(" | Train Loss: "); Serial.print(train_loss, 4);
    Serial.print(" | Val Loss: "); Serial.println(val_loss, 4);
  }
}

// Fine-tune Classifier
void finetuneLabeled(const char *path) {
  int total = countLines(path);
  int split = total * TRAIN_RATIO;

  for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
    File file = SD.open(path);
    if (!file) {
      Serial.println("Failed to open labeled training file.");
      return;
    }

    file.readStringUntil('\n'); // skip header
    int line_num = 0;
    while (file.available() && line_num < split) {
      String line = file.readStringUntil('\n');
      char buf[200];
      line.toCharArray(buf, 200);
      char *token = strtok(buf, ",");

      genann_type inputs[INPUTS];
      genann_type expected[OUTPUTS] = {0};
      for (int i = 0; i < INPUTS; i++) {
        inputs[i] = atof(token);
        token = strtok(NULL, ",");
      }

      int label = atoi(token);
      expected[label] = 1.0;
      genann_train(ann, inputs, expected, 0.1);
      line_num++;
    }
    file.close();

    float train_acc, train_loss, val_acc, val_loss;
    evaluateFile(path, 0, split, true, train_acc, train_loss);
    evaluateFile(path, split, total, true, val_acc, val_loss);

    Serial.print("Fine-tune Epoch "); Serial.print(epoch + 1);
    Serial.print(" | Train Acc: "); Serial.print(train_acc, 2);
    Serial.print("%, Loss: "); Serial.print(train_loss, 4);
    Serial.print(" | Val Acc: "); Serial.print(val_acc, 2);
    Serial.print("%, Loss: "); Serial.println(val_loss, 4);
  }
}

// Test on test_labeled.csv
void testModel(const char *path) {
  File file = SD.open(path);
  if (!file) {
    Serial.println("Test file not found.");
    return;
  }

  file.readStringUntil('\n'); // skip header
  Serial.println("Test Results (first 10):");
  int test_count = 0;
  while (file.available() && test_count < 10) {
    String line = file.readStringUntil('\n');
    char buf[200];
    line.toCharArray(buf, 200);
    char *token = strtok(buf, ",");

    genann_type inputs[INPUTS];
    for (int i = 0; i < INPUTS; i++) {
      inputs[i] = atof(token);
      token = strtok(NULL, ",");
    }
    int actual_label = atoi(token);

    const genann_type *output = genann_run(ann, inputs);
    int predicted_label = 0;
    genann_type max_out = output[0];
    for (int j = 1; j < OUTPUTS; j++) {
      if (output[j] > max_out) {
        max_out = output[j];
        predicted_label = j;
      }
    }

    Serial.print("Sample "); Serial.print(test_count + 1);
    Serial.print(" | Actual: "); Serial.print(actual_label);
    Serial.print(" | Predicted: "); Serial.println(predicted_label);
    test_count++;
  }

  file.close();
}

// Main setup
void setup() {
  Serial.begin(115200);
  if (!SD.begin(CS_PIN)) {
    Serial.println("SD card init failed!");
    return;
  }

  ann = genann_init(INPUTS, 1, HIDDEN, OUTPUTS);

  trainAutoencoder("/dataset/health/train_unlabeled.csv");
  finetuneLabeled("/dataset/health/train_labeled.csv");
  testModel("/dataset/health/test_labeled.csv");

  File modelFile = SD.open("/dataset/health/model_ssl.txt", FILE_WRITE);
  if (modelFile) {
    for (int i = 0; i < ann->total_weights; i++) {
      modelFile.println(ann->weight[i], 6);
    }
    modelFile.close();
    Serial.println("Model saved.");
  }
}

void loop() {
  // Nothing here
}
