import ee
import pandas as pd
import joblib
import requests
import sys
import os
from datetime import datetime, timedelta

# --- KONFIGURASI UTAMA (WAJIB DIISI) ---
MODEL_PATH = 'rf_pm25_model_bundle.joblib' 
URL_TARGET_API = os.environ.get('URL_TARGET_API') 
LOKASI_NAMA = 'Depok_Test'
LOKASI_KOORDINAT = [106.821389, -6.398983] # [lon, lat]

# !! PENTING !!
# Ganti ini agar sesuai dengan SEMUA fitur yang dibutuhkan model Anda
GEE_FEATURE_CONFIG = {
    # 'nama_fitur_di_model': ('NAMA_DATASET_GEE', 'NAMA_BAND', SKALA_METER),
    'NO2_tropo': ('COPERNICUS/S5P/OFFL/L3_NO2', 'tropospheric_NO2_column_number_density', 1000),
    'T2m_C': ('ECMWF/ERA5_LAND/HOURLY', 'temperature_2m', 11132),
    
    # 'CO': ('COPERNICUS/S5P/OFFL/L3_CO', 'CO_column_number_density', 1000),
    # 'O3': ('COPERNICUS/S5P/OFFL/L3_O3', 'O3_column_number_density', 1000),
    # 'SO2': ('COPERNICUS/S5P/OFFL/L3_SO2', 'SO2_column_number_density', 1000),
    # 'WindSpeed': ('ECMWF/ERA5_LAND/HOURLY', 'wind_speed_10m', 11132),
    # 'RH': ('ECMWF/ERA5_LAND/HOURLY', 'relative_humidity_2m', 11132),
    # ... Tambahkan SEMUA fitur Anda di sini ...
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
            
            # Penanganan khusus untuk band suhu (Kelvin -> Celsius)
            if band_name == 'temperature_2m':
                def k_to_c(image):
                    return image.subtract(273.15).copyProperties(image, image.propertyNames())
                image = collection.map(k_to_c).mean().clip(aoi)
            # Penanganan khusus untuk kelembaban (dihitung dari T dan Td)
            elif band_name == 'relative_humidity_2m':
                rh_coll = ee.ImageCollection(collection_name).filterDate(start_str, end_str).select(['temperature_2m', 'dewpoint_temperature_2m'])
                def calc_rh(image):
                    t = image.select('temperature_2m').subtract(273.15) # K ke C
                    td = image.select('dewpoint_temperature_2m').subtract(273.15) # K ke C
                    rh = td.divide(t).multiply(100) # Rumus sederhana, GANTI DENGAN RUMUS ANDA
                    return rh.rename('relative_humidity_2m')
                image = rh_coll.map(calc_rh).mean().clip(aoi)
            else:
                image = collection.mean().clip(aoi)
            
            value = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi, scale=scale).get(band_name).getInfo()
            if value is not None:
                print(f"  ✓ Ditemukan data {band_name} pada: {start_str} (Nilai: {value:.2f})")
                return value, start_str
        except Exception as e:
            # Sering terjadi jika tidak ada gambar sama sekali di hari itu
            pass 
    print(f"  ✗ GAGAL: Tidak ditemukan data {band_name} setelah {MAX_LOOKBACK_DAYS} hari.")
    return None, start_date.strftime('%Y-%m-%d')

def get_features_from_gee(aoi, date):
    """Mengambil SEMUA data fitur dari GEE dengan logika ffill."""
    feature_dict = {}
    feature_dates = {}
    for feature_name, (coll, band, scale) in GEE_FEATURE_CONFIG.items():
        value, data_date = get_latest_non_null_value(coll, band, aoi, scale, date)
        feature_dict[feature_name] = value if value is not None else 0
        feature_dates[feature_name] = data_date
    print("✓ Pengambilan data GEE selesai.")
    return feature_dict, feature_dates

# --- FUNGSI UTAMA (MAIN) ---
print("=" * 60)
print(f"Memulai GitHub Action: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

if URL_TARGET_API is None:
    print("✗ GAGAL: Secret 'URL_TARGET_API' tidak ditemukan.")
    print("  Pastikan Anda sudah menambahkannya di Settings -> Secrets -> Actions.")
    sys.exit(1)

# 1. Inisialisasi GEE (SEKARANG AKAN BERHASIL)
try:
    # ee.Initialize() akan otomatis menemukan env var GOOGLE_APPLICATION_CREDENTIALS
    # yang diatur di file .yml
    ee.Initialize() 
    print("✓ Otentikasi GEE berhasil (via Service Account).")
except Exception as e:
    print(f"✗ GAGAL Otentikasi GEE: {e}")
    print("  Pastikan secret 'GEE_SERVICE_ACCOUNT_KEY' sudah benar dan .yml sudah di-update.")
    sys.exit(1)

# 2. Muat Model
try:
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    feature_cols = model_bundle['feature_cols']
    print(f"✓ Model '{MODEL_PATH}' berhasil dimuat.")
    print(f"  Model ini membutuhkan {len(feature_cols)} fitur: {feature_cols}")
except Exception as e:
    print(f"✗ GAGAL memuat model: {e}")
    sys.exit(1)

# 3. Ambil Data GEE
print("\n--- Mengambil Fitur dari GEE (dengan ffill) ---")
target_date = datetime.now() # Ambil data paling update s/d "hari ini"
aoi = ee.Geometry.Point(LOKASI_KOORDINAT)
data_fitur, data_dates = get_features_from_gee(aoi, target_date)
print(f"Fitur yang didapat: {data_fitur}")

# 4. Lakukan Estimasi
print("\n--- Melakukan Estimasi PM2.5 ---")
try:
    # Buat DataFrame HANYA dengan fitur yang didapat dari GEE
    df_fitur = pd.DataFrame([data_fitur])
    
    # PERIKSA KESESUAIAN FITUR
    missing_in_model = set(df_fitur.columns) - set(feature_cols)
    if missing_in_model:
        print(f"  WARNING: Fitur GEE ada yg tidak dipakai model: {missing_in_model}")

    missing_in_gee = set(feature_cols) - set(df_fitur.columns)
    if missing_in_gee:
        print(f"  ✗ GAGAL: Fitur yang dibutuhkan model TIDAK ADA di GEE_FEATURE_CONFIG:")
        print(f"    {missing_in_gee}")
        print("    Pastikan 'GEE_FEATURE_CONFIG' di main.py sudah lengkap.")
        sys.exit(1)
        
    # Susun ulang kolom DataFrame agar SAMA PERSIS dengan urutan 'feature_cols'
    df_fitur_ordered = df_fitur[feature_cols] 
    
    # Scaling
    data_scaled = scaler.transform(df_fitur_ordered)
    
    # Prediksi
    prediksi_pm25 = model.predict(data_scaled)[0]
    print(f"✓✓ HASIL ESTIMASI: {prediksi_pm25:.4f} µg/m³")
    
except Exception as e:
    print(f"✗ GAGAL saat estimasi: {e}")
    print("  Ini bisa terjadi jika urutan fitur tidak cocok, atau data GEE null.")
    sys.exit(1)

# 5. Kirim Hasil ke API Anda
print("\n--- Mengirim Hasil ke API Target ---")
payload = {
    'lokasi': LOKASI_NAMA,
    'tanggal_prediksi': target_date.strftime('%Y-%m-%d'),
    'prediksi_pm25': prediksi_pm25,
    'sumber_data': 'GEE_RF_Model_GitHub_Action',
    'tanggal_fitur': data_dates, # Kirim info tanggal asli datanya
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
