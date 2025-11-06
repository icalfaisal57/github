"""
Script untuk debug dan inspect file GeoJSON
Jalankan: python debug_geojson.py
"""

import json
import sys

GEOJSON_PATH = 'depok_32748.geojson'
KOLOM_KECAMATAN = 'WADMKC'

def inspect_geojson():
    """Inspect struktur GeoJSON dan tampilkan informasi detail."""
    
    print("=" * 70)
    print("🔍 DEBUG GEOJSON FILE")
    print("=" * 70)
    
    try:
        with open(GEOJSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n✓ File berhasil dimuat: {GEOJSON_PATH}")
        print(f"  Type: {data.get('type')}")
        print(f"  Total features: {len(data.get('features', []))}")
        
        # Cek CRS
        if 'crs' in data:
            print(f"  CRS: {data['crs']}")
        
        # Analisis properties
        print("\n" + "=" * 70)
        print("📊 ANALISIS PROPERTIES")
        print("=" * 70)
        
        if data.get('features'):
            first_feature = data['features'][0]
            props = first_feature.get('properties', {})
            
            print(f"\nContoh properties dari feature pertama:")
            for key, value in props.items():
                print(f"  {key}: {value}")
            
            print(f"\n✓ Kolom '{KOLOM_KECAMATAN}' ditemukan: {KOLOM_KECAMATAN in props}")
            
            # Hitung unique kecamatan
            kecamatan_count = {}
            for feature in data['features']:
                nama = feature['properties'].get(KOLOM_KECAMATAN, 'Unknown')
                kecamatan_count[nama] = kecamatan_count.get(nama, 0) + 1
            
            print("\n" + "=" * 70)
            print("📍 DISTRIBUSI KECAMATAN")
            print("=" * 70)
            print(f"\nTotal kecamatan unik: {len(kecamatan_count)}")
            print("\nJumlah feature per kecamatan:")
            for nama, count in sorted(kecamatan_count.items()):
                print(f"  {nama:20s}: {count:3d} features")
        
        # Analisis geometry
        print("\n" + "=" * 70)
        print("🗺️  ANALISIS GEOMETRY")
        print("=" * 70)
        
        geometry_types = {}
        coord_depths = {}
        
        for idx, feature in enumerate(data['features'][:5]):  # Sample 5 pertama
            geom = feature.get('geometry', {})
            geom_type = geom.get('type')
            coords = geom.get('coordinates', [])
            
            geometry_types[geom_type] = geometry_types.get(geom_type, 0) + 1
            
            depth = get_depth(coords)
            coord_depths[depth] = coord_depths.get(depth, 0) + 1
            
            nama = feature['properties'].get(KOLOM_KECAMATAN, 'Unknown')
            
            print(f"\nFeature {idx + 1}: {nama}")
            print(f"  Geometry type: {geom_type}")
            print(f"  Coordinate depth: {depth}")
            print(f"  Sample coords: {str(coords)[:150]}...")
            
            # Cek apakah koordinat valid (lon: 106-107, lat: -7 sampai -6)
            if coords:
                sample_coord = get_first_coordinate(coords)
                if sample_coord:
                    if len(sample_coord) == 3:
                        lon, lat, z = sample_coord
                        print(f"  First coordinate: [{lon:.4f}, {lat:.4f}, {z:.1f}]")
                        print(f"  ⚠ PERINGATAN: Koordinat 3D terdeteksi!")
                        print(f"    GEE hanya terima 2D (lon, lat)")
                        print(f"    → Kode sudah menangani ini otomatis (hapus Z)")
                    elif len(sample_coord) == 2:
                        lon, lat = sample_coord
                        print(f"  First coordinate: [{lon:.4f}, {lat:.4f}]")
                        print(f"  ✓ Koordinat 2D (sudah benar)")
                    
                    lon = sample_coord[0]
                    lat = sample_coord[1]
                    
                    valid_lon = 106 <= lon <= 107
                    valid_lat = -7 <= lat <= -6
                    
                    if valid_lon and valid_lat:
                        print(f"  ✓ Koordinat valid untuk Depok")
                    else:
                        print(f"  ⚠ PERINGATAN: Koordinat di luar Depok!")
                        print(f"    Expected: lon ≈ 106.8, lat ≈ -6.4")
        
        print("\n" + "=" * 70)
        print("📊 RINGKASAN GEOMETRY TYPES")
        print("=" * 70)
        for geom_type, count in geometry_types.items():
            print(f"  {geom_type}: {count}")
        
        print("\n" + "=" * 70)
        print("💡 REKOMENDASI")
        print("=" * 70)
        
        # Berikan rekomendasi
        if len(kecamatan_count) == 11:
            print("✓ Jumlah kecamatan sudah benar (11 kecamatan di Depok)")
        elif len(kecamatan_count) > 11:
            print(f"⚠ Ada {len(kecamatan_count)} features (lebih dari 11 kecamatan)")
            print("  Kemungkinan:")
            print("  1. Data berisi kelurahan/desa (bukan kecamatan)")
            print("  2. Geometri terpecah per bagian")
            print("  → Kode sudah menangani ini dengan merge otomatis")
        
        # Cek geometry type dominan
        dominant_type = max(geometry_types, key=geometry_types.get)
        if dominant_type == 'Polygon':
            print("✓ Geometry type dominan: Polygon (OK)")
        elif dominant_type == 'MultiPolygon':
            print("✓ Geometry type dominan: MultiPolygon (OK)")
        else:
            print(f"⚠ Geometry type tidak standar: {dominant_type}")
        
        print("\n" + "=" * 70)
        print("✅ INSPECT SELESAI")
        print("=" * 70)
        
        return True
        
    except FileNotFoundError:
        print(f"\n✗ File tidak ditemukan: {GEOJSON_PATH}")
        print("  Pastikan file ada di direktori yang sama dengan script ini.")
        return False
    except json.JSONDecodeError as e:
        print(f"\n✗ File GeoJSON tidak valid: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_depth(obj, current_depth=0):
    """Hitung kedalaman nested list."""
    if not isinstance(obj, list) or len(obj) == 0:
        return current_depth
    if isinstance(obj[0], (int, float)):
        return current_depth
    return get_depth(obj[0], current_depth + 1)

def get_first_coordinate(coords):
    """Ekstrak koordinat pertama dari nested list."""
    if not isinstance(coords, list) or len(coords) == 0:
        return None
    if isinstance(coords[0], (int, float)) and len(coords) >= 2:
        return coords[0], coords[1]
    return get_first_coordinate(coords[0])

if __name__ == '__main__':
    success = inspect_geojson()
    sys.exit(0 if success else 1)
