#!/usr/bin/env python3
"""
Script to compile Kubeflow pipeline to YAML format
"""

import sys
import os

# Add current directory to path to import the pipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Output to parent dir as model-retraining.yaml (DSP/OpenShift AI pipeline name)
_output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
package_path = os.path.join(_output_dir, "model-retraining.yaml")

try:
    import kfp
    from battery_ml_pipeline import battery_ml_pipeline
    
    print("Compiling Battery ML pipeline...")
    
    # Compile pipeline to YAML
    kfp.compiler.Compiler().compile(
        pipeline_func=battery_ml_pipeline,
        package_path=package_path
    )
    
    print(f"Pipeline compiled successfully to '{package_path}'")
    print("\nInstructions for using in OpenShift AI:")
    print("1. Upload the 'model-retraining.yaml' file to your OpenShift AI instance")
    print("2. Go to Data Science Pipelines > Pipelines")
    print("3. Click on 'Import pipeline' and select the model-retraining.yaml file")
    print("4. Create a new pipeline run with appropriate parameters:")
    print("   - influxdb_url: URL of your InfluxDB")
    print("   - influxdb_token: Access token")
    print("   - aws_s3_endpoint: MinIO/S3 endpoint")
    print("   - aws_access_key_id: Access key")
    print("   - aws_secret_access_key: Secret key")
    print("5. Monitor execution from the web interface")
    
    # Display pipeline information
    print(f"\nPipeline information:")
    print(f"- Name: battery-ml-pipeline")
    print(f"- Components: 8 (retrieve -> prepare -> train_stress -> train_ttf -> validate_stress -> validate_ttf -> save_stress -> save_ttf)")
    print(f"- Notebooks included: 01, 02, 03, 05, 07")
    print(f"- Generated file: {os.path.abspath(package_path)}")
    
except ImportError as e:
    print(f"Error: Could not import kfp. Install with: pip install kfp")
    print(f"Details: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error compiling pipeline: {e}")
    sys.exit(1)
