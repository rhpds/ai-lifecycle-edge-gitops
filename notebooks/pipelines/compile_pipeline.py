#!/usr/bin/env python3
"""
Script para compilar la pipeline de Kubeflow a formato YAML
"""

import sys
import os

# Añadir el directorio actual al path para importar la pipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import kfp
    from battery_ml_pipeline import battery_ml_pipeline
    
    print("Compilando pipeline de Battery ML...")
    
    # Compilar la pipeline a YAML
    kfp.compiler.Compiler().compile(
        pipeline_func=battery_ml_pipeline,
        package_path="battery_ml_pipeline.yaml"
    )
    
    print("✅ Pipeline compilada exitosamente a 'battery_ml_pipeline.yaml'")
    print("\n📋 Instrucciones para usar en OpenShift AI:")
    print("1. Sube el archivo 'battery_ml_pipeline.yaml' a tu instancia de OpenShift AI")
    print("2. Ve a Data Science Pipelines > Pipelines")
    print("3. Haz clic en 'Import pipeline' y selecciona el archivo YAML")
    print("4. Crea una nueva pipeline run con los parámetros apropiados:")
    print("   - influxdb_url: URL de tu InfluxDB")
    print("   - influxdb_token: Token de acceso")
    print("   - aws_s3_endpoint: Endpoint de tu MinIO/S3")
    print("   - aws_access_key_id: Clave de acceso")
    print("   - aws_secret_access_key: Clave secreta")
    print("5. Monitorea la ejecución desde la interfaz web")
    
    # Mostrar información sobre la pipeline
    print(f"\n📊 Información de la pipeline:")
    print(f"- Nombre: battery-ml-pipeline")
    print(f"- Componentes: 7 (retrieve → prepare → train_stress → predict_stress → train_ttf → predict_ttf → save_s3)")
    print(f"- Notebooks incluidos: 01, 02, 03, 04, 05, 06, 07")
    print(f"- Archivo generado: {os.path.abspath('battery_ml_pipeline.yaml')}")
    
except ImportError as e:
    print(f"❌ Error: No se pudo importar kfp. Instala con: pip install kfp")
    print(f"Detalles: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error compilando la pipeline: {e}")
    sys.exit(1)
