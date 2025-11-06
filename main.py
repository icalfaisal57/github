import ee
import pandas as pd
import joblib
import requests
import sys
import os
import math
import json
from datetime import datetime, timedelta

# --- KONFIGURASI UTAMA (WAJIB DIISI) ---
MODEL_PATH = 'rf_pm25_model_bundle.joblib' 
URL_TARGET_API = os.environ.get('URL_TARGET_API') 
GEOJSON_PATH = 'depok_32748.geojson'  # Path ke file GeoJSON
KOTA_NAMA = 'Depok'
KOLOM_KECAMATAN = 'WADMKC'  # Nama kolom kecamatan di GeoJSON
DEBUG_MODE = True  # Set True untuk melihat detail geometry

# !! PENTING !!
# Semua fitur yang dibutuhkan model
GEE_FEATURE_CONFIG = {
    'NO2_tropo': ('COPERNICUS/S5P/OFFL/L3_NO2', 'tropospheric_NO2_column_number_density', 1000),
    'T2m_C': ('ECMWF/ERA5_LAND/HOURLY', 'temperature_2m', 11132),
    'CO': ('COPERNICUS/S5P/OFFL/L3_CO', 'CO_column_number_density', 1000),
    'O3': ('COPERNICUS/S5P/OFFL/L3_O3', 'O3_column_number_density', 1000),
    'SO2': ('COPERNICUS/S5P/OFFL/L3_SO2', 'SO2_column_number_density', 1000),
    'AOD': ('MODIS/061/MCD19A2_GRANULES', 'Optical_Depth_055', 1000),
}

MAX_LOOKBACK_DAYS = 30 # Batas pencarian data mundur
# ----------------------------------------

def load_geojson_boundaries():
    """Memuat file GeoJSON dan mengekstrak boundaries per kecamatan."""
    try:
        with open(GEOJSON_PATH, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        
        print(f"✓ File GeoJSON '{GEOJSON_PATH}' berhasil dimuat.")
        print(f"  Total features dalam file: {len(geojson_data['features'])}")
        
        # Dictionary untuk menggabungkan geometri dengan nama kecamatan yang sama
        kecamatan_dict = {}
        
        for feature in geojson_data['features']:
            props = feature['properties']
            geometry = feature['geometry']
            
            kecamatan_nama = props.get(KOLOM_KECAMATAN, 'Unknown')
            
            # Skip jika geometry tidak valid
            if not geometry or not geometry.get('coordinates'):
                print(f"  ⚠ Skipping {kecamatan_nama}: No coordinates")
                continue
            
            # Gabungkan geometri untuk kecamatan yang sama
            if kecamatan_nama not in kecamatan_dict:
                kecamatan_dict[kecamatan_nama] = {
                    'nama': kecamatan_nama,
                    'geometries': [],
                    'properties': props
                }
            
            kecamatan_dict[kecamatan_nama]['geometries'].append(geometry)
        
        # Konversi dictionary ke list dan gabungkan geometri
        kecamatan_list = []
        for nama, data in kecamatan_dict.items():
            # Jika ada multiple geometries untuk satu kecamatan, gabungkan jadi MultiPolygon
            if len(data['geometries']) > 1:
                all_coords = []
                for geom in data['geometries']:
                    if geom['type'] == 'Polygon':
                        all_coords.append(geom['coordinates'])
                    elif geom['type'] == 'MultiPolygon':
                        all_coords.extend(geom['coordinates'])
                
                merged_geometry = {
                    'type': 'MultiPolygon',
                    'coordinates': all_coords
                }
            else:
                merged_geometry = data['geometries'][0]
            
            kecamatan_list.append({
                'nama': nama,
                'geometry': merged_geometry,
                'properties': data['properties']
            })
        
        print(f"✓ Ditemukan {len(kecamatan_list)} kecamatan unik:")
        for kec in sorted(kecamatan_list, key=lambda x: x['nama']):
            geom_type = kec['geometry']['type']
            print(f"  - {kec['nama']:20s} ({geom_type})")
        
        return kecamatan_list
    
    except FileNotFoundError:
        print(f"✗ GAGAL: File '{GEOJSON_PATH}' tidak ditemukan.")
        print("  Pastikan file GeoJSON ada di root direktori repositori.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"✗ GAGAL: File GeoJSON tidak valid: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ GAGAL memuat GeoJSON: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def geojson_to_ee_geometry(geojson_geometry):
    """Konversi GeoJSON geometry ke ee.Geometry dengan validasi."""
    try:
        geom_type = geojson_geometry['type']
        coords = geojson_geometry['coordinates']
        
        # Validasi koordinat tidak kosong
        if not coords:
            print(f"  ✗ Koordinat kosong")
            return None
        
        if geom_type == 'Polygon':
            # Polygon: [[[lon, lat], [lon, lat], ...]]
            # Pastikan format benar
            if not isinstance(coords, list) or len(coords) == 0:
                print(f"  ✗ Format Polygon tidak valid")
                return None
            
            # Validasi setiap koordinat
            try:
                # GEE expects: [[[lon, lat], ...]]
                return ee.Geometry.Polygon(coords)
            except Exception as e:
                print(f"  ✗ Error create Polygon: {str(e)[:100]}")
                return None
                
        elif geom_type == 'MultiPolygon':
            # MultiPolygon: [[[[lon, lat], ...]], [[[lon, lat], ...]]]
            if not isinstance(coords, list) or len(coords) == 0:
                print(f"  ✗ Format MultiPolygon tidak valid")
                return None
            
            try:
                # GEE expects list of polygon coordinates
                return ee.Geometry.MultiPolygon(coords)
            except Exception as e:
                print(f"  ✗ Error create MultiPolygon: {str(e)[:100]}")
                # Coba fallback: buat dari polygon pertama saja
                try:
                    print(f"  ℹ Mencoba fallback: gunakan polygon pertama saja...")
                    return ee.Geometry.Polygon(coords[0])
                except:
                    return None
        
        else:
            print(f"  ✗ Geometry type '{geom_type}' tidak didukung")
            return None
            
    except KeyError as e:
        print(f"  ✗ Missing key in geometry: {e}")
        return None
    except Exception as e:
        print(f"  ✗ Error konversi geometry: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        return None

def get_latest_non_null_value(collection_name, band_name, aoi, scale, start_date):
    """Mencari mundur (ffill) untuk mendapatkan nilai non-null terakhir dari GEE."""
    print(f"    Mencari {band_name}...")
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
            
            value = image.reduceRegion(
                reducer=ee.Reducer.mean(), 
                geometry=aoi, 
                scale=scale,
                maxPixels=1e9
            ).get(band_name).getInfo()
            
            if value is not None:
                print(f"    ✓ Data {band_name}: {start_str} (Nilai: {value:.6e})")
                return value, start_str
        except Exception as e:
            # print(f"      Error pada {start_str}: {str(e)[:50]}")
            pass 
    print(f"    ✗ Tidak ditemukan data {band_name} dalam {MAX_LOOKBACK_DAYS} hari.")
    return None, start_date.strftime('%Y-%m-%d')

def get_wind_features(aoi, scale, start_date):
    """Menghitung Wind Speed dan Wind Direction dari komponen U dan V."""
    print(f"    Mencari data angin...")
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
                scale=scale,
                maxPixels=1e9
            ).get('u_component_of_wind_10m').getInfo()
            
            v_value = v_image.reduceRegion(
                reducer=ee.Reducer.mean(), 
                geometry=aoi, 
                scale=scale,
                maxPixels=1e9
            ).get('v_component_of_wind_10m').getInfo()
            
            if u_value is not None and v_value is not None:
                # Hitung Wind Speed dan Direction
                wind_speed = math.sqrt(u_value**2 + v_value**2)
                wind_direction = math.degrees(math.atan2(v_value, u_value)) % 360
                
                print(f"    ✓ Data angin: {start_str} (Speed: {wind_speed:.2f} m/s, Dir: {wind_direction:.1f}°)")
                return wind_speed, wind_direction, start_str
        except Exception as e:
            # print(f"      Error pada {start_str}: {str(e)[:50]}")
            pass
    
    print(f"    ✗ Tidak ditemukan data angin dalam {MAX_LOOKBACK_DAYS} hari.")
    return 0, 0, start_date.strftime('%Y-%m-%d')

def get_features_from_gee(aoi, date, kecamatan_nama):
    """Mengambil SEMUA data fitur dari GEE dengan logika ffill untuk satu kecamatan."""
    feature_dict = {}
    feature_dates = {}
    
    # Daftar fitur yang tidak boleh negatif
    NON_NEGATIVE_FEATURES = ['CO', 'NO2_tropo', 'O3', 'SO2', 'AOD', 'Wind_Speed']
    
    print(f"  📊 Mengambil data untuk: {kecamatan_nama}")
    
    # Ambil fitur reguler dari GEE_FEATURE_CONFIG
    for feature_name, (coll, band, scale) in GEE_FEATURE_CONFIG.items():
        value, data_date = get_latest_non_null_value(coll, band, aoi, scale, date)
        
        if value is None:
            print(f"    ⚠ '{feature_name}' diisi 0 (tidak ada data)")
            feature_dict[feature_name] = 0
        else:
            # Handle nilai negatif untuk polutan
            if feature_name in NON_NEGATIVE_FEATURES and value < 0:
                print(f"    ⚠ {feature_name} negatif ({value:.6e}) → dikoreksi ke 0")
                feature_dict[feature_name] = 0
            else:
                feature_dict[feature_name] = value
        
        feature_dates[feature_name] = data_date
    
    # Ambil data angin
    wind_speed, wind_dir, wind_date = get_wind_features(aoi, 11132, date)
    
    if wind_speed < 0:
        print(f"    ⚠ Wind Speed negatif ({wind_speed:.2f}) → dikoreksi ke 0")
        wind_speed = 0
    
    feature_dict['Wind_Speed'] = wind_speed
    feature_dict['Wind_Direction'] = wind_dir
    feature_dates['Wind_Speed'] = wind_date
    feature_dates['Wind_Direction'] = wind_date

    print(f"  ✓ Selesai mengambil {len(feature_dict)} fitur untuk {kecamatan_nama}")
    return feature_dict, feature_dates

def estimate_pm25_for_kecamatan(kecamatan_data, model, scaler, feature_cols, target_date):
    """Estimasi PM2.5 untuk satu kecamatan."""
    kecamatan_nama = kecamatan_data['nama']
    
    print(f"\n{'='*60}")
    print(f"🏘️  KECAMATAN: {kecamatan_nama.upper()}")
    print(f"{'='*60}")
    
    # Konversi GeoJSON geometry ke ee.Geometry
    ee_geometry = geojson_to_ee_geometry(kecamatan_data['geometry'])
    
    if ee_geometry is None:
        print(f"  ✗ GAGAL: Tidak bisa konversi geometry untuk {kecamatan_nama}")
        return None
    
    # Ambil fitur dari GEE
    try:
        data_fitur, data_dates = get_features_from_gee(ee_geometry, target_date, kecamatan_nama)
    except Exception as e:
        print(f"  ✗ GAGAL mengambil data GEE: {e}")
        return None
    
    # Lakukan estimasi
    try:
        df_fitur = pd.DataFrame([data_fitur])
        
        missing_features = set(feature_cols) - set(df_fitur.columns)
        if missing_features:
            print(f"  ✗ GAGAL: Fitur tidak lengkap: {missing_features}")
            return None
        
        df_fitur_ordered = df_fitur[feature_cols]
        data_scaled = scaler.transform(df_fitur_ordered)
        prediksi_pm25 = model.predict(data_scaled)[0]
        
        print(f"  ✅ HASIL ESTIMASI: {prediksi_pm25:.2f} µg/m³")
        
        return {
            'kecamatan': kecamatan_nama,
            'kota': KOTA_NAMA,
            'tanggal_prediksi': target_date.strftime('%Y-%m-%d'),
            'prediksi_pm25': float(prediksi_pm25),
            'sumber_data': 'GEE_RF_Model_GitHub_Action',
            'tanggal_fitur': data_dates,
            'fitur_asli': {k: float(v) for k, v in data_fitur.items()},
            'properties': kecamatan_data['properties']
        }
    
    except Exception as e:
        print(f"  ✗ GAGAL saat estimasi: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==============================================================================
# --- FUNGSI UTAMA (MAIN) ---
# ==============================================================================
print("=" * 70)
print(f"🚀 ESTIMASI PM2.5 PER KECAMATAN - KOTA DEPOK")
print(f"   Waktu Mulai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

if URL_TARGET_API is None:
    print("✗ GAGAL: Secret 'URL_TARGET_API' tidak ditemukan.")
    print("  Pastikan sudah menambahkannya di Settings → Secrets → Actions.")
    sys.exit(1)

# --- 1. Inisialisasi GEE ---
try:
    SERVICE_ACCOUNT_EMAIL = 'estimatepm25@tugas-akhir-473911.iam.gserviceaccount.com'
    KEY_FILE_PATH = 'service_key.json'
    
    print(f"\n🔐 Autentikasi Google Earth Engine...")
    print(f"   Akun: {SERVICE_ACCOUNT_EMAIL}")
    
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT_EMAIL, KEY_FILE_PATH)
    ee.Initialize(credentials)
    
    print("   ✓ Autentikasi GEE berhasil!")

except Exception as e:
    print(f"✗ GAGAL Autentikasi GEE: {e}")
    sys.exit(1)

# --- 2. Muat Model ---
try:
    print(f"\n🤖 Memuat Model Machine Learning...")
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    feature_cols = model_bundle['feature_cols']
    
    print(f"   ✓ Model berhasil dimuat")
    print(f"   Fitur yang dibutuhkan ({len(feature_cols)}): {feature_cols}")
except Exception as e:
    print(f"✗ GAGAL memuat model: {e}")
    sys.exit(1)

# --- 3. Load GeoJSON Boundaries ---
print(f"\n🗺️  Memuat Batas Wilayah Kecamatan...")
kecamatan_list = load_geojson_boundaries()

# --- 4. Estimasi PM2.5 untuk Setiap Kecamatan ---
print(f"\n{'='*70}")
print(f"📍 MEMULAI ESTIMASI UNTUK {len(kecamatan_list)} KECAMATAN")
print(f"{'='*70}")

target_date = datetime.now()
results = []
success_count = 0
failed_count = 0

for idx, kecamatan_data in enumerate(kecamatan_list, 1):
    print(f"\n[{idx}/{len(kecamatan_list)}] Processing...")
    
    result = estimate_pm25_for_kecamatan(
        kecamatan_data, 
        model, 
        scaler, 
        feature_cols, 
        target_date
    )
    
    if result:
        results.append(result)
        success_count += 1
    else:
        failed_count += 1

# --- 5. Ringkasan Hasil ---
print(f"\n{'='*70}")
print(f"📊 RINGKASAN HASIL ESTIMASI")
print(f"{'='*70}")
print(f"✅ Berhasil: {success_count}/{len(kecamatan_list)} kecamatan")
print(f"❌ Gagal   : {failed_count}/{len(kecamatan_list)} kecamatan")

if results:
    print(f"\n{'Kecamatan':<25} {'PM2.5 (µg/m³)':>15}")
    print("-" * 42)
    for r in sorted(results, key=lambda x: x['prediksi_pm25'], reverse=True):
        print(f"{r['kecamatan']:<25} {r['prediksi_pm25']:>15.2f}")
    
    avg_pm25 = sum(r['prediksi_pm25'] for r in results) / len(results)
    print("-" * 42)
    print(f"{'RATA-RATA KOTA DEPOK':<25} {avg_pm25:>15.2f}")

# --- 6. Kirim Hasil ke API ---
if results and URL_TARGET_API:
    print(f"\n{'='*70}")
    print(f"📤 MENGIRIM DATA KE API")
    print(f"{'='*70}")
    
    for idx, result in enumerate(results, 1):
        print(f"\n[{idx}/{len(results)}] Mengirim data {result['kecamatan']}...")
        
        try:
            response = requests.post(URL_TARGET_API, json=result, timeout=15)
            
            if 200 <= response.status_code < 300:
                print(f"   ✓ Berhasil (Status: {response.status_code})")
            else:
                print(f"   ✗ Gagal (Status: {response.status_code})")
                print(f"   Response: {response.text[:100]}")
        
        except requests.exceptions.ConnectionError:
            print(f"   ✗ Gagal koneksi ke API")
        except Exception as e:
            print(f"   ✗ Error: {e}")

print(f"\n{'='*70}")
print(f"✅ PROSES SELESAI")
print(f"   Waktu Selesai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*70}")
