const int numSamples = 300;
int ecgData[numSamples];
const int ecgPin = A0;
bool dataCollected = false;  // Flag untuk menandai data sudah diambil

void setup() {
  Serial.begin(9600);
}

void loop() {
  // Hanya mengambil data jika belum pernah diambil
  if (!dataCollected) {
    // Mengambil 300 sampel
    for(int n = 0; n < numSamples; n++) {
      ecgData[n] = analogRead(ecgPin);
      // Cetak data langsung setelah 
      Serial.println(ecgData[n]);
      delay(10);
    }
    dataCollected = true;  // Set flag bahwa data sudah diambil
    Serial.println("Data collection completed");
  }
  // Program akan tetap berjalan tapi tidak mengambil data baru
}
