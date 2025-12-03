import serial

class SerialManager:
    _instance = None

    def __new__(cls, port="COM3", baud=115200):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            try:
                cls._instance.serial = serial.Serial(port, baud, timeout=1)
            except Exception as e:
                print("⚠ No se pudo abrir el puerto serial:", e)
                cls._instance.serial = None
        return cls._instance

    def write(self, data):
        if self.serial and self.serial.is_open:
            self.serial.write(data)

    def close(self):
        if self.serial and self.serial.is_open:
            self.serial.close()
