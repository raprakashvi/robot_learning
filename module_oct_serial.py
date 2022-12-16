


import serial.tools.list_ports
import serial
import time

class octserial():

    def __init__(self):
        """ Initialize serial commands for the OCT
        """
        self.cmd_volscan = "StartCapture VolumeScan"
        self.cmd_saveprocessedimages = "SaveQueueImages"
        self.cmd_saverawimages = "SaveRawQueueImages"
        self.cmd_samplename = "SetSampleName"

    def portlist(self):
        """ Lists all the active ports on the connected computer
        """
        ports = serial.tools.list_ports.comports()

        for port, desc, hwid in sorted(ports):
                print("{}: {} [{}]".format(port, desc, hwid))

    def portconfigure(self):
        """ Configures serial port for accessing the OCT
        """
        ser = serial.Serial()
        ser.baudrate = 115200
        ser.port = 'COM3'

        ser.open()
        if(ser.is_open):
            print("{} port open".format(ser.port))
            ser.close()
            return 1
        else:
            print("check port connection")
            return 0 

    def octsave(self, ser):
        """ Saves the scanned file
        """
        ser.open()
        scan_save_encode = self.cmd_saveprocessedimages.encode()
        # save the scanned file
        ser.write(scan_save_encode + b'\x0a')
        print("Save command {}".format(scan_save_encode + b'\x0a'))
        serial_output = ser.read_until(expected= '\n', size = 23)
    
        print("Output recived: {}".format(serial_output))
        if serial_output == ("SaveQueueImagesComplete".encode()):
            print("Images save complete")
            ser.close()
            return 1
        else:
            print("Image save Failed")
            ser.close()
            return 0




    def octvolscan(self,file_name):
        """ Send commands to capture volumetric scan from the OCT 
        """
        time_start = time.clock()
        while self.portconfigure() :

            ser = serial.Serial()
            ser.baudrate = 115200
            ser.port = 'COM3'
            ser.parity = serial.PARITY_NONE
            ser.bytesize = serial.EIGHTBITS
            ser.stopbits = serial.STOPBITS_ONE
            ser.timeout = 30
            
            if ser.isOpen():
                ser.close()
            ser.open()
            token = 0
            
            #with ser as s:
            # set file name before scan
            print("writing sample name")
            sample_name_encode = (self.cmd_samplename + file_name).encode()
            ser.write(sample_name_encode + b'\x0a')
            print("Write command sent {}".format(sample_name_encode  + b'\x0a'))
            
            # give command for volumetric scan
            print("starting volumetric scan")
            vol_scan_encode = self.cmd_volscan.encode()
            ser.write(vol_scan_encode + b'\x0a')
            print("Write command sent {}".format(vol_scan_encode + b'\x0a'))
            serial_output = ser.read_until(expected= '\n',size = 15) 
            print("Vol scan output {}".format(serial_output))
            if serial_output == ("CaptureComplete".encode()):
                print("Volumteric Scan complete")
                token = 1
            else:
                print("Volumetric scan failed")
            ser.close()
            octsave_token = self.octsave(ser)
            time_end = time.clock() - time_start
            print("Total time to scan = {}".format(time_end))
            if octsave_token == 1:
                return 1
            else :
                return 0
                
        

#%%

if __name__ == "__main__" :

    oct = octserial()

    #test 1
    # Test configuration
    oct.portconfigure()

    #test 2
    # Test write
    oct.octvolscan("test")
    

            

               


