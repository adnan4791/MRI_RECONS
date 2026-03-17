import h5py
import numpy as np
import os

def convert_h5_to_cfl():
    # 1. Input nama file dari user
    input_h5 = input("Masukkan nama file H5 (contoh: data.h5): ")
    
    if not os.path.exists(input_h5):
        print(f"Error: File '{input_h5}' tidak ditemukan.")
        return

    output_prefix = input("Masukkan nama output (tanpa ekstensi): ")

    try:
        # 2. Membuka file H5
        with h5py.File(input_h5, 'r') as f:
            print("Membaca struktur file...")
            
            # Cek struktur file - bisa ISMRMRD standar atau struktur kustom
            if 'dataset' in f and 'data' in f['dataset']:
                # Struktur ISMRMRD standar
                data = f['dataset']['data'][:]
                print(f"Data ISMRMRD standar ditemukan dengan shape: {data.shape}")
            elif 'kspace' in f:
                # Struktur kustom dengan key 'kspace'
                data = f['kspace'][:]
                print(f"Data k-space ditemukan dengan shape: {data.shape}")
            else:
                print("Error: Struktur data tidak dikenali.")
                print("Daftar key yang tersedia:", list(f.keys()))
                return

            # 3. Konversi ke Complex64 (Format standar BART/CFL)
            # Pastikan data dalam bentuk complex float 32-bit
            data_cfl = data.astype(np.complex64)

            # 4. Menulis file .cfl (Binary data)
            cfl_filename = f"{output_prefix}.cfl"
            data_cfl.tofile(cfl_filename)
            print(f"Berhasil membuat: {cfl_filename}")

            # 5. Menulis file .hdr (Header ASCII)
            hdr_filename = f"{output_prefix}.hdr"
            with open(hdr_filename, 'w') as hdr_file:
                # Header format BART sederhana
                hdr_file.write("# BART v0.4.04\n")
                # Tulis dimensi dipisahkan spasi
                dims_str = " ".join(map(str, data.shape))
                hdr_file.write(dims_str + " \n")
            
            print(f"Berhasil membuat: {hdr_filename}")
            print("\nKonversi Selesai!")

    except Exception as e:
        print(f"Terjadi kesalahan saat memproses file: {e}")

if __name__ == "__main__":
    convert_h5_to_cfl()