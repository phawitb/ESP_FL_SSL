#include <SPI.h>
#include <SD.h>
#include "genann.h"

#define INPUTS 9
#define HIDDEN 9
#define OUTPUTS 7   // updated for label range 0–6
#define MAX_EPOCHS 10
#define TRAIN_RATIO 0.8

#if defined(ESP32)
  #define CS_PIN 5
#elif defined(ESP8266)
  #define CS_PIN 4
#else
  #define CS_PIN 10  // fallback for other boards
#endif

genann *ann;
File dataFile, testFile;

// Count number of lines in CSV
int countLines(const char *path) {
  File file = SD.open(path);
  if (!file) return 0;

  int count = 0;
  while (file.available()) {
    file.readStringUntil('\n');
    count++;
  }
  file.close();
  return count;
}

// Evaluate set (streaming version)
void evaluateFile(const char *path, int startLine, int endLine, float &acc, float &loss) {
  File file = SD.open(path);
  if (!file) {
    acc = 0; loss = 0;
    return;
  }

  int correct = 0;
  float total_loss = 0;
  int line_num = 0;
  while (file.available()) {
    String line = file.readStringUntil('\n');
    if (line_num >= startLine && line_num < endLine) {
      genann_type inputs[INPUTS];
      int label;
      char buf[200];
      line.toCharArray(buf, 200);
      char *token = strtok(buf, ",");

      for (int i = 0; i < INPUTS; i++) {
        inputs[i] = atof(token);
        token = strtok(NULL, ",");
      }
      label = atoi(token);  // no "-1"

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
    }
    line_num++;
  }

  file.close();
  int total = endLine - startLine;
  acc = 100.0 * correct / total;
  loss = total_loss / total;
}

void setup() {
  Serial.begin(115200);
  if (!SD.begin(CS_PIN)) {
    Serial.println("SD card init failed!");
    return;
  }

  ann = genann_init(INPUTS, 1, HIDDEN, OUTPUTS);

  const char *trainPath = "/dataset/health/train0.csv";
  int total_lines = countLines(trainPath);
  int train_lines = (int)(total_lines * TRAIN_RATIO);

  for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
    dataFile = SD.open(trainPath);
    if (!dataFile) {
      Serial.println("Failed to open training file.");
      return;
    }

    int line_num = 0;
    while (dataFile.available()) {
      String line = dataFile.readStringUntil('\n');
      if (line_num >= train_lines) break;

      genann_type inputs[INPUTS];
      genann_type expected[OUTPUTS] = {0};
      char buf[200];
      line.toCharArray(buf, 200);
      char *token = strtok(buf, ",");

      for (int i = 0; i < INPUTS; i++) {
        inputs[i] = atof(token);
        token = strtok(NULL, ",");
      }
      int label = atoi(token);  // no "-1"
      expected[label] = 1.0;

      genann_train(ann, inputs, expected, 0.1);
      line_num++;
    }
    dataFile.close();

    float train_acc, train_loss, val_acc, val_loss;
    evaluateFile(trainPath, 0, train_lines, train_acc, train_loss);
    evaluateFile(trainPath, train_lines, total_lines, val_acc, val_loss);

    Serial.print("Epoch "); Serial.print(epoch + 1);
    Serial.print(" | Train Acc: "); Serial.print(train_acc, 2);
    Serial.print("%, Loss: "); Serial.print(train_loss, 4);
    Serial.print(" | Val Acc: "); Serial.print(val_acc, 2);
    Serial.print("%, Loss: "); Serial.println(val_loss, 4);
  }

  // Save model
  File modelFile = SD.open("/dataset/health/model.txt", FILE_WRITE);
  if (modelFile) {
    for (int i = 0; i < ann->total_weights; i++) {
      modelFile.println(ann->weight[i], 6);
    }
    modelFile.close();
    Serial.println("Model saved.");
  }

  // Evaluate on test.csv
  testFile = SD.open("/dataset/health/test.csv");
  if (!testFile) {
    Serial.println("Test file not found.");
    return;
  }

  Serial.println("Test Results (first 10):");
  int test_count = 0;
  while (testFile.available() && test_count < 20) {
    String line = testFile.readStringUntil('\n');
    char buf[200];
    line.toCharArray(buf, 200);

    genann_type inputs[INPUTS];
    int actual_label = 0;

    char *token = strtok(buf, ",");
    for (int i = 0; i < INPUTS; i++) {
      inputs[i] = atof(token);
      token = strtok(NULL, ",");
    }
    actual_label = atoi(token);  // no "-1"

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

  testFile.close();
}

void loop() {
  // Nothing here
}