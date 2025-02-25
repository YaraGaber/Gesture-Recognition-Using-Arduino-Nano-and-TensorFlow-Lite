def tflite_to_header(tflite_path, header_path):                       
    with open(tflite_path, 'rb') as tflite_file:
        tflite_data = tflite_file.read()
    
    with open(header_path, 'w') as header_file:
        header_file.write('#ifndef MODEL_H\n')
        header_file.write('#define MODEL_H\n\n')
        header_file.write('const unsigned char g_model[] = {\n')
        
        # Write data as hex values
        for i, byte in enumerate(tflite_data):
            header_file.write(f'0x{byte:02x}, ')
            if (i + 1) % 12 == 0:  # 12 bytes per line for readability
                header_file.write('\n')
        
        header_file.write('\n};\n\n')
        header_file.write(f'const unsigned int g_model_len = {len(tflite_data)};\n\n')
        header_file.write('#endif // MODEL_H\n')


# Example usage
tflite_to_header(
    r'C:\Users\yarag\OneDrive\Documents\doucments arduino\codes\adv1\gesture_recognition_model_improved.tflite',
    r'C:\Users\yarag\OneDrive\Documents\doucments arduino\codes\adv1\model.h'
)
