import ee
import pandas as pd
import joblib
import requests
import sys
import os
import math
from datetime import datetime, timedelta

GEE_FEATURE_CONFIG = {
    'NO2_tropo': ('COPERNICUS/S5P/OFFL/L3_NO2', 'tropospheric_NO2_column_number_density', 1000),
    'T2m_C': ('ECMWF/ERA5_LAND/HOURLY', 'temperature_2m', 11132),
    'CO': ('COPERNICUS/S5P/OFFL/L3_CO', 'CO_column_number_density', 1000),
    'O3': ('COPERNICUS/S5P/OFFL/L3_O3', 'O3_column_number_density', 1000),
    'SO2': ('COPERNICUS/S5P/OFFL/L3_SO2', 'SO2_column_number_density', 1000),
    'AOD': ('MODIS/006/MCD19A2_GRANULES', 'Optical_Depth_047', 1000),
}

# --- KONFIGURASI UTAMA (WAJIB DIISI) ---
MODEL_PATH = 'rf_pm25_model_bundle.joblib' 
URL_TARGET_API = os.environ.get('URL_TARGET_API') 
LOKASI_NAMA = 'Depok_Test'
LOKASI_KOORDINAT = [106.821389, -6.398983] # [lon, lat]

# !! PENTING !!
# Ganti/tambahkan di sini agar sesuai dengan SEMUA fitur yang dibutuhkan model Anda
# Format: 'nama_fitur_di_model': ('NAMA_DATASET_GEE', 'NAMA_BAND', SKALA_METER),
GEE_FEATURE_CONFIG = {
    'NO2_tropo': ('COPERNICUS/S5P/OFFL/L3_NO2', 'tropospheric_NO2_column_number_density', 1000),
    'T2m_C': ('ECMWF/ERA5_LAND/HOURLY', 'temperature_2m', 11132),
    'CO': ('COPERNICUS/S5P/OFFL/L3_CO', 'CO_column_number_density', 1000),
    'O3': ('COPERNICUS/S5P/OFFL/L3_O3', 'O3_column_number_density', 1000),
    'SO2': ('COPERNICUS/S5P/OFFL/L3_SO2', 'SO2_column_number_density', 1000),
    'AOD': ('MODIS/006/MCD19A2_GRANULES', 'Optical_Depth_047', 1000),
}

MAX_LOOKBACK_DAYS = 30 # Batas pencarian data mundur
# ----------------------------------------

def get_latest_non_null_value(collection_name, band_name, aoi, scale, start_date):
    """Mencari mundur (ffill) untuk mendapatkan nilai non-null terakhir dari GEE."""
    print(f"  Mencari {band_name}...")
    for i in range(MAX_LOOKBACK_DAYS):
        current_date = start_date - timedelta(days=i)
        start_str = current_date.strftime('%Y-%m-%d')
        end_str = (current_date + timedelta(days=1)).strftime('%Y-%m-%d')
        try:
            collection = ee.ImageCollection(collection_name).filterDate(start_str, end_str).select(band_name)
            
            if band_name == 'temperature_2m':
                def k_to_c(image):
                    return image.subtract(273.15).copyProperties(image, image.propertyNames())
                image = collection.map(k_to_c).mean().clip(aoi)
            else:
                image = collection.mean().clip(aoi)
            
            value = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi, scale=scale).get(band_name).getInfo()
            
            if value is not None:
                print(f"  ✓ Ditemukan data {band_name} pada: {start_str} (Nilai: {value:.2f})")
                return value, start_str
        except Exception as e:
            pass 
    print(f"  ✗ GAGAL: Tidak ditemukan data {band_name} setelah {MAX_LOOKBACK_DAYS} hari.")
    return None, start_date.strftime('%Y-%m-%d')

def get_wind_features(aoi, scale, start_date):
    """Menghitung Wind Speed dan Wind Direction dari komponen U dan V."""
    print(f"  Mencari data angin (Wind Speed & Direction)...")
    for i in range(MAX_LOOKBACK_DAYS):
        current_date = start_date - timedelta(days=i)
        start_str = current_date.strftime('%Y-%m-%d')
        end_str = (current_date + timedelta(days=1)).strftime('%Y-%m-%d')
        try:
            collection = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').filterDate(start_str, end_str)
            
            # Ambil komponen U dan V
            u_image = collection.select('u_component_of_wind_10m').mean().clip(aoi)
            v_image = collection.select('v_component_of_wind_10m').mean().clip(aoi)
            
            u_value = u_image.reduceRegion(
                reducer=ee.Reducer.mean(), 
                geometry=aoi, 
                scale=scale
            ).get('u_component_of_wind_10m').getInfo()
            
            v_value = v_image.reduceRegion(
                reducer=ee.Reducer.mean(), 
                geometry=aoi, 
                scale=scale
            ).get('v_component_of_wind_10m').getInfo()
            
            if u_value is not None and v_value is not None:
                # Hitung Wind Speed dan Direction
                wind_speed = math.sqrt(u_value**2 + v_value**2)
                wind_direction = math.degrees(math.atan2(v_value, u_value)) % 360
                
                print(f"  ✓ Ditemukan data angin pada: {start_str}")
                print(f"    Wind Speed: {wind_speed:.2f} m/s, Wind Direction: {wind_direction:.2f}°")
                return wind_speed, wind_direction, start_str
        except Exception as e:
            pass
    
    print(f"  ✗ GAGAL: Tidak ditemukan data angin setelah {MAX_LOOKBACK_DAYS} hari.")
    return 0, 0, start_date.strftime('%Y-%m-%d')

def get_features_from_gee(aoi, date):
    """Mengambil SEMUA data fitur dari GEE dengan logika ffill."""
    feature_dict = {}
    feature_dates = {} 
    
    # Ambil fitur reguler dari GEE_FEATURE_CONFIG
    for feature_name, (coll, band, scale) in GEE_FEATURE_CONFIG.items():
        value, data_date = get_latest_non_null_value(coll, band, aoi, scale, date)
        
        if value is None:
            print(f"  WARNING: '{feature_name}' diisi 0 setelah gagal mencari.")
            feature_dict[feature_name] = 0
        else:
            feature_dict[feature_name] = value
        
        feature_dates[feature_name] = data_date
    
    # Ambil data angin (Wind Speed & Direction) secara terpisah
    wind_speed, wind_dir, wind_date = get_wind_features(aoi, 11132, date)
    feature_dict['Wind_Speed'] = wind_speed
    feature_dict['Wind_Direction'] = wind_dir
    feature_dates['Wind_Speed'] = wind_date
    feature_dates['Wind_Direction'] = wind_date

    print("✓ Pengambilan data GEE selesai.")
    return feature_dict, feature_dates

# ==============================================================================
# --- FUNGSI UTAMA (MAIN) ---
# ==============================================================================
print("=" * 60)
print(f"Memulai GitHub Action: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

if URL_TARGET_API is None:
    print("✗ GAGAL: Secret 'URL_TARGET_API' tidak ditemukan.")
    print("  Pastikan Anda sudah menambahkannya di Settings -> Secrets -> Actions.")
    sys.exit(1)

# --- 1. Inisialisasi GEE (Metode Eksplisit) ---
try:
    # Info ini harus sesuai dengan file JSON Anda
    SERVICE_ACCOUNT_EMAIL = 'estimatepm25@tugas-akhir-473911.iam.gserviceaccount.com'
    
    # Path ini harus sesuai dengan yang dibuat di file .yml
    KEY_FILE_PATH = 'service_key.json' 
    
    print(f"Mencoba autentikasi GEE secara EKSPLISIT dengan:")
    print(f"  Akun: {SERVICE_ACCOUNT_EMAIL}")
    print(f"  File Kunci: {KEY_FILE_PATH}")
    
    # Buat objek kredensial secara manual
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT_EMAIL, KEY_FILE_PATH)
    
    # Inisialisasi GEE dengan kredensial tersebut
    ee.Initialize(credentials)
    
    print("✓ Otentikasi GEE berhasil (Metode Eksplisit).")

except Exception as e:
    print(f"✗ GAGAL Otentikasi GEE (Metode Eksplisit): {e}")
    print("  Ini bisa terjadi jika:")
    print("  1. Isi GEE_SERVICE_ACCOUNT_KEY salah.")
    print("  2. Akun Layanan tidak memiliki peran 'Editor' di IAM Google Cloud.")
    print("  3. Email di SERVICE_ACCOUNT_EMAIL salah.")
    sys.exit(1)

# --- 2. Muat Model ---
try:
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    feature_cols = model_bundle['feature_cols'] # Daftar fitur yang dibutuhkan model
    
    print(f"✓ Model '{MODEL_PATH}' berhasil dimuat.")
    print(f"  Model ini membutuhkan {len(feature_cols)} fitur:")
    print(f"  {feature_cols}")
except Exception as e:
    print(f"✗ GAGAL memuat model: {e}")
    print(f"  Pastikan file '{MODEL_PATH}' ada di repositori dan dilacak LFS.")
    sys.exit(1)

# --- 3. Ambil Data Fitur dari GEE ---
print("\n--- Mengambil Fitur dari GEE (dengan ffill) ---")
target_date = datetime.now() 
aoi = ee.Geometry.Point(LOKASI_KOORDINAT)
data_fitur, data_dates = get_features_from_gee(aoi, target_date)
print(f"Fitur yang didapat: {data_fitur}")

# --- 4. Lakukan Estimasi ---
print("\n--- Melakukan Estimasi PM2.5 ---")
try:
    df_fitur = pd.DataFrame([data_fitur])
    
    missing_in_gee = set(feature_cols) - set(df_fitur.columns)
    if missing_in_gee:
        print(f"  ✗ GAGAL: Fitur yang dibutuhkan model TIDAK ADA di GEE_FEATURE_CONFIG:")
        print(f"    {missing_in_gee}")
        print("    Harap tambahkan fitur ini ke 'GEE_FEATURE_CONFIG' di main.py.")
        sys.exit(1)
        
    df_fitur_ordered = df_fitur[feature_cols] 
    data_scaled = scaler.transform(df_fitur_ordered)
    prediksi_pm25 = model.predict(data_scaled)[0]
    print(f"✓✓ HASIL ESTIMASI: {prediksi_pm25:.4f} µg/m³")
    
except Exception as e:
    print(f"✗ GAGAL saat estimasi: {e}")
    sys.exit(1)

# --- 5. Kirim Hasil ke API Anda ---
print("\n--- Mengirim Hasil ke API Target ---")
payload = {
    'lokasi': LOKASI_NAMA,
    'tanggal_prediksi': target_date.strftime('%Y-%m-%d'),
    'prediksi_pm25': prediksi_pm25,
    'sumber_data': 'GEE_RF_Model_GitHub_Action',
    'tanggal_fitur': data_dates, 
    'fitur_asli': data_fitur
}

try:
    response = requests.post(URL_TARGET_API, json=payload, timeout=15)
    if 200 <= response.status_code < 300:
        print(f"✓ Berhasil mengirim data ke API (Webhook.site). Status: {response.status_code}")
    else:
        print(f"✗ Gagal mengirim data ke API. Status: {response.status_code}")
        print(f"  Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print(f"✗ GAGAL: Koneksi ke API gagal. Periksa URL: {URL_TARGET_API}")
except Exception as e:
    print(f"✗ GAGAL koneksi API: {e}")

print("\n" + "=" * 60)
print("Proses GitHub Action Selesai.")
print("=" * 60)
