import ee
import pandas as pd
import joblib
import requests
import sys
import os # Import os untuk membaca 'Secrets'
from datetime import datetime, timedelta

# --- KONFIGURASI UTAMA ---
# Path model SAMA DENGAN nama file di repo
MODEL_PATH = 'rf_pm25_model_bundle.joblib' 
# Baca URL API dari GitHub Secret
URL_TARGET_API = os.environ.get('URL_TARGET_API') 

LOKASI_NAMA = 'Depok_Test'
LOKASI_KOORDINAT = [106.821389, -6.398983] # [lon, lat]

# Konfigurasi GEE (SAMA SEPERTI SEBELUMNYA)
# (Dataset, Nama Band, Skala Reduksi)
GEE_FEATURE_CONFIG = {
    'NO2_tropo': ('COPERN.../L3_NO2', 'tropospheric_NO2...', 1000),
    'T2m_C': ('ECMWF/ERA5_LAND/HOURLY', 'temperature_2m', 11132),
    # ... Tambahkan SEMUA fitur Anda di sini ...
}
MAX_LOOKBACK_DAYS = 30
# ----------------------------------------

# Fungsi ffill GEE (SAMA SEPERTI SEBELUMNYA)
def get_latest_non_null_value(collection_name, band_name, aoi, scale, start_date):
    print(f"  Mencari {band_name}...")
    for i in range(MAX_LOOKBACK_DAYS):
        current_date = start_date - timedelta(days=i)
        start_str = current_date.strftime('%Y-%m-%d')
        end_str = (current_date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        try:
            collection = ee.ImageCollection(collection_name) \
                           .filterDate(start_str, end_str) \
                           .select(band_name)
            
            if band_name == 'temperature_2m': # Konversi K ke C
                def k_to_c(image):
                    return image.subtract(273.15).copyProperties(image, image.propertyNames())
                image = collection.map(k_to_c).mean().clip(aoi)
            else:
                image = collection.mean().clip(aoi)

            value = image.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=aoi, scale=scale
            ).get(band_name).getInfo()

            if value is not None:
                print(f"  ✓ Ditemukan data {band_name} pada: {start_str}")
                return value, start_str
        except Exception:
            pass 
    print(f"  ✗ GAGAL: Tidak ditemukan data {band_name} setelah {MAX_LOOKBACK_DAYS} hari.")
    return None, start_date.strftime('%Y-%m-%d')

# Fungsi Pengambil Fitur GEE (SAMA)
def get_features_from_gee(aoi, date):
    feature_dict = {}
    feature_dates = {}
    for feature_name, (coll, band, scale) in GEE_FEATURE_CONFIG.items():
        value, data_date = get_latest_non_null_value(coll, band, aoi, scale, date)
        feature_dict[feature_name] = value if value is not None else 0
        feature_dates[feature_name] = data_date
    return feature_dict, feature_dates

# --- FUNGSI UTAMA (MAIN) ---
# Tidak perlu 'def main():', kita jalankan langsung
print("=" * 60)
print(f"Memulai GitHub Action: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# Cek apakah 'Secret' ada
if URL_TARGET_API is None:
    print("✗ GAGAL: Secret 'URL_TARGET_API' tidak ditemukan.")
    print("  Pastikan Anda sudah menambahkannya di Settings -> Secrets -> Actions.")
    sys.exit(1) # Keluar dengan error

# 1. Inisialisasi GEE
try:
    # Autentikasi akan ditangani oleh file credentials dari GitHub Secret
    ee.Initialize()
    print("✓ Otentikasi GEE berhasil.")
except Exception as e:
    print(f"✗ GAGAL Otentikasi GEE: {e}")
    print("  Pastikan secret 'GEE_CREDENTIALS' sudah benar.")
    sys.exit(1)

# 2. Muat Model
try:
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    feature_cols = model_bundle['feature_cols']
    print(f"✓ Model '{MODEL_PATH}' berhasil dimuat.")
except Exception as e:
    print(f"✗ GAGAL memuat model: {e}")
    sys.exit(1)

# 3. Ambil Data GEE
print("\n--- Mengambil Fitur dari GEE (dengan ffill) ---")
target_date = datetime.now()
aoi = ee.Geometry.Point(LOKASI_KOORDINAT)
data_fitur, data_dates = get_features_from_gee(aoi, target_date)
print(f"Fitur yang didapat: {data_fitur}")

# 4. Lakukan Estimasi
print("\n--- Melakukan Estimasi PM2.5 ---")
try:
    df_fitur = pd.DataFrame([data_fitur])[feature_cols] # Susun ulang kolom
    data_scaled = scaler.transform(df_fitur)
    prediksi_pm25 = model.predict(data_scaled)[0]
    print(f"✓✓ HASIL ESTIMASI: {prediksi_pm25:.4f} µg/m³")
except Exception as e:
    print(f"✗ GAGAL saat estimasi: {e}")
    sys.exit(1)

# 5. Kirim Hasil ke API
print("\n--- Mengirim Hasil ke API Target ---")
payload = {
    'lokasi': LOKASI_NAMA,
    'tanggal_prediksi': target_date.strftime('%Y-%m-%d'),
    'prediksi_pm25': prediksi_pm25,
    'sumber_data': 'GEE_RF_Model_GitHub_Action',
    'tanggal_fitur': data_dates
}

try:
    response = requests.post(URL_TARGET_API, json=payload, timeout=15)
    if 200 <= response.status_code < 300:
        print(f"✓ Berhasil mengirim data ke API. Status: {response.status_code}")
    else:
        print(f"✗ Gagal mengirim data ke API. Status: {response.status_code}")
        print(f"  Response: {response.text}")
except Exception as e:
    print(f"✗ GAGAL koneksi API: {e}")

print("\nProses GitHub Action Selesai.")