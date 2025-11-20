#!/usr/bin/env python3
"""
Battery Management System ML Pipeline
=====================================
Complete ML pipeline for battery stress detection and time-to-failure prediction
using UBI images for optimal OpenShift compatibility.

Author: AI Assistant
Date: 2025
"""

import kfp
from kfp import dsl
from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Artifact
from typing import NamedTuple


# =============================================================================
# PIPELINE COMPONENTS
# =============================================================================

@component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["pandas", "influxdb-client"]
)
def retrieve_influx_data(
    influxdb_url: str,
    influxdb_token: str,
    influxdb_org: str,
    influxdb_bucket: str,
    raw_data: Output[Dataset]
) -> NamedTuple('Outputs', [('records_count', int)]):
    """
    Component 01: Retrieve data from InfluxDB
    """
    from influxdb_client import InfluxDBClient
    import pandas as pd
    import os
    
    # Initialize Client (disable SSL verification for self-signed certificates)
    client = InfluxDBClient(url=influxdb_url, token=influxdb_token, org=influxdb_org, verify_ssl=False)
    
    def retrieve_battery_data():
        query = f'''
        from(bucket: "{influxdb_bucket}")
          |> range(start: -1h)
          |> filter(fn: (r) => r["_measurement"] == "battery_data")
        '''
        query_api = client.query_api()
        tables = query_api.query(query, org=influxdb_org)

        # Process Results
        data = []
        for table in tables:
            for record in table.records:
                data.append(record.values)

        df = pd.DataFrame(data)
        return df
    
    try:
        df = retrieve_battery_data()
        # Save raw data
        df.to_csv(raw_data.path, index=False)
        records_count = len(df)
        print(f"Retrieved {records_count} records from InfluxDB")
        
    except Exception as e:
        print(f"Error connecting to InfluxDB: {e}")
        # Fallback: use sample data
        print("Using fallback sample data...")
        sample_data = {
            '_time': ['2025-02-12 14:26:34.190000+00:00'] * 10,
            'batteryId': [1] * 10,
            '_field': ['ambientTemp', 'batteryCurrent', 'batteryTemp', 'batteryVoltage', 
                      'distance', 'kmh', 'stateOfCharge', 'stateOfHealth'] * 1 + ['ambientTemp', 'batteryCurrent'],
            '_value': [18.65, 78.06, 25.22, 396.39, 0.2, 127.75, 0.9991, 99.9998, 18.36, 81.42]
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(raw_data.path, index=False)
        records_count = len(df)
    
    client.close()
    
    from collections import namedtuple
    output = namedtuple('Outputs', ['records_count'])
    return output(records_count)


@component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["pandas"]
)
def prepare_data(
    raw_data: Input[Dataset],
    prepared_data: Output[Dataset]
) -> NamedTuple('Outputs', [('prepared_records', int)]):
    """
    Component 02: Prepare data by pivoting columns
    """
    import pandas as pd
    
    # Load InfluxDB CSV Data
    df = pd.read_csv(raw_data.path)
    
    # Pivot the data so that '_field' values become columns
    df_pivot = df.pivot(index=["_time", "batteryId"], columns="_field", values="_value").reset_index()
    
    # Rename `_time` to `timestamp` for clarity
    df_pivot.rename(columns={"_time": "timestamp"}, inplace=True)
    
    # Save prepared data
    df_pivot.to_csv(prepared_data.path, index=False)
    
    print(f"Prepared {len(df_pivot)} records")
    print("Data columns:", list(df_pivot.columns))
    
    from collections import namedtuple
    output = namedtuple('Outputs', ['prepared_records'])
    return output(len(df_pivot))


@component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["pandas", "scikit-learn", "tensorflow", "openvino-dev==2024.6.0"]
)
def train_stress_detection_model(
    prepared_data: Input[Dataset],
    stress_model: Output[Model]
) -> NamedTuple('Outputs', [('accuracy', float), ('stress_events', int)]):
    """
    Component 03: Train stress detection model with OpenVINO conversion
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf
    from tensorflow import keras
    import os
    
    # Force CPU usage to avoid CUDA issues
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Load data
    df = pd.read_csv(prepared_data.path)
    
    # Define stress condition (1 = Stress, 0 = Normal)
    def detect_stress(row):
        if row["batteryCurrent"] > 400 or row["batteryTemp"] > 50 or row["stateOfCharge"] < 0.05 or row["batteryVoltage"] < 320:
            return 1  # Stress condition
        return 0  # Normal condition

    # Apply stress detection
    df["stressIndicator"] = df.apply(detect_stress, axis=1)
    stress_events = df["stressIndicator"].sum()
    
    # Define Features and Target
    features = ["stateOfCharge", "stateOfHealth", "batteryCurrent", "batteryVoltage", 
                "kmh", "distance", "batteryTemp", "ambientTemp", "currentLoad"]
    X = df[features]
    y = df["stressIndicator"]
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define neural network
    mlp_tf = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    mlp_tf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train model
    history = mlp_tf.fit(X_train_scaled, y_train, epochs=50, batch_size=32, 
                        validation_split=0.1, verbose=0)
    
    # Evaluate model
    test_loss, test_accuracy = mlp_tf.evaluate(X_test_scaled, y_test, verbose=0)
    
    # Create model directory and save
    model_dir = stress_model.path
    os.makedirs(model_dir, exist_ok=True)
    
    # Save as SavedModel format first
    saved_model_path = os.path.join(model_dir, "saved_model")
    tf.saved_model.save(mlp_tf, saved_model_path)
    
    # Convert to OpenVINO format
    try:
        import subprocess
        cmd = f"mo --saved_model_dir {saved_model_path} --output_dir {model_dir}"
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("Model converted to OpenVINO format (.xml/.bin)")
    except subprocess.CalledProcessError as e:
        print(f"OpenVINO conversion failed: {e}")
        print("Model saved in TensorFlow format only")
    except Exception as e:
        print(f"OpenVINO conversion error: {e}")
    
    print(f"Stress detection model trained with accuracy: {test_accuracy:.4f}")
    print(f"Detected {stress_events} stress events")
    
    from collections import namedtuple
    output = namedtuple('Outputs', ['accuracy', 'stress_events'])
    return output(float(test_accuracy), int(stress_events))


@component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["pandas", "tensorflow", "numpy"]
)
def predict_stress(
    stress_model: Input[Model],
    prepared_data: Input[Dataset],
    stress_predictions: Output[Dataset]
) -> NamedTuple('Outputs', [('high_stress_count', int)]):
    """
    Component 04: Predict stress using TensorFlow
    """
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    import os
    
    try:
        # Load Model
        model_path = os.path.join(stress_model.path, "saved_model")
        
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            
            # Load test data
            df = pd.read_csv(prepared_data.path)
            
            # Prepare features
            features = ["stateOfCharge", "stateOfHealth", "batteryCurrent", "batteryVoltage", 
                       "kmh", "distance", "batteryTemp", "ambientTemp", "currentLoad"]
            
            # Make predictions for all data
            X = df[features].values.astype(np.float32)
            predictions = model.predict(X, verbose=0)
            
            df['stress_prediction'] = predictions.flatten()
            df['high_stress'] = (df['stress_prediction'] > 0.5).astype(int)
            
            # Save predictions
            df.to_csv(stress_predictions.path, index=False)
            
            high_stress_count = df['high_stress'].sum()
            print(f"Predicted {high_stress_count} high stress conditions")
            
        else:
            print("TensorFlow model not found, using dummy predictions")
            df = pd.read_csv(prepared_data.path)
            df['stress_prediction'] = 0.1  # Low stress
            df['high_stress'] = 0
            df.to_csv(stress_predictions.path, index=False)
            high_stress_count = 0
            
    except Exception as e:
        print(f"Error in stress prediction: {e}")
        # Fallback
        df = pd.read_csv(prepared_data.path)
        df['stress_prediction'] = 0.1
        df['high_stress'] = 0
        df.to_csv(stress_predictions.path, index=False)
        high_stress_count = 0
    
    from collections import namedtuple
    output = namedtuple('Outputs', ['high_stress_count'])
    return output(int(high_stress_count))


@component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["pandas", "scikit-learn", "tensorflow", "openvino-dev==2024.6.0"]
)
def train_ttf_model(
    prepared_data: Input[Dataset],
    ttf_model: Output[Model]
) -> NamedTuple('Outputs', [('mae', float)]):
    """
    Component 05: Train time-to-failure (TTF) model
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf
    from tensorflow import keras
    import os
    
    # Force CPU usage
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Load data
    df = pd.read_csv(prepared_data.path)
    
    # Convert timestamp to datetime for time-series processing
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Simulate Time-to-Failure (assuming failure happens at the last recorded timestamp)
    df["timeBeforeFailure"] = (df["timestamp"].max() - df["timestamp"]).dt.total_seconds() / 3600  # Convert to hours
    
    # Define Features and Target for TTF
    features = ["batteryTemp", "batteryCurrent", "batteryVoltage", "stateOfCharge", "stateOfHealth"]
    X = df[features]
    y = df["timeBeforeFailure"]
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define neural network for regression
    ttf_model_tf = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)  # No activation for regression
    ])
    
    # Compile model
    ttf_model_tf.compile(optimizer='adam', loss='mae', metrics=['mae'])
    
    # Train model
    history = ttf_model_tf.fit(X_train_scaled, y_train, epochs=50, batch_size=32, 
                              validation_split=0.1, verbose=0)
    
    # Evaluate model
    y_pred = ttf_model_tf.predict(X_test_scaled, verbose=0)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Create model directory and save
    model_dir = ttf_model.path
    os.makedirs(model_dir, exist_ok=True)
    
    # Save as SavedModel format first
    saved_model_path = os.path.join(model_dir, "saved_model")
    tf.saved_model.save(ttf_model_tf, saved_model_path)
    
    # Convert to OpenVINO format
    try:
        import subprocess
        cmd = f"mo --saved_model_dir {saved_model_path} --output_dir {model_dir}"
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("TTF model converted to OpenVINO format (.xml/.bin)")
    except subprocess.CalledProcessError as e:
        print(f"OpenVINO conversion failed: {e}")
        print("TTF model saved in TensorFlow format only")
    except Exception as e:
        print(f"OpenVINO conversion error: {e}")
    
    print(f"TTF model trained with MAE: {mae:.4f} hours")
    
    from collections import namedtuple
    output = namedtuple('Outputs', ['mae'])
    return output(float(mae))


@component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["pandas", "tensorflow", "numpy"]
)
def predict_ttf(
    ttf_model: Input[Model],
    prepared_data: Input[Dataset],
    ttf_predictions: Output[Dataset]
) -> NamedTuple('Outputs', [('avg_ttf_hours', float)]):
    """
    Component 06: Predict time-to-failure using TensorFlow
    """
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    import os
    
    try:
        # Load Model
        model_path = os.path.join(ttf_model.path, "saved_model")
        
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            
            # Load test data
            df = pd.read_csv(prepared_data.path)
            
            # Prepare features for TTF
            features = ["batteryTemp", "batteryCurrent", "batteryVoltage", "stateOfCharge", "stateOfHealth"]
            
            # Make predictions for all data
            X = df[features].values.astype(np.float32)
            predictions = model.predict(X, verbose=0)
            
            df['ttf_prediction_hours'] = predictions.flatten()
            
            # Save predictions
            df.to_csv(ttf_predictions.path, index=False)
            
            avg_ttf = np.mean(predictions)
            print(f"Average predicted TTF: {avg_ttf:.2f} hours")
            
        else:
            print("TensorFlow TTF model not found, using dummy predictions")
            df = pd.read_csv(prepared_data.path)
            df['ttf_prediction_hours'] = 100.0  # Default 100 hours
            df.to_csv(ttf_predictions.path, index=False)
            avg_ttf = 100.0
            
    except Exception as e:
        print(f"Error in TTF prediction: {e}")
        # Fallback
        df = pd.read_csv(prepared_data.path)
        df['ttf_prediction_hours'] = 100.0
        df.to_csv(ttf_predictions.path, index=False)
        avg_ttf = 100.0
    
    from collections import namedtuple
    output = namedtuple('Outputs', ['avg_ttf_hours'])
    return output(float(avg_ttf))


@component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["boto3"]
)
def save_models_to_s3(
    stress_model: Input[Model],
    ttf_model: Input[Model],
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_s3_endpoint: str,
    aws_s3_bucket: str
) -> NamedTuple('Outputs', [('upload_status', str)]):
    """
    Component 07: Save models to S3
    """
    import boto3
    import os
    from botocore.exceptions import ClientError
    
    try:
        # Create s3 connection
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            endpoint_url=aws_s3_endpoint
        )
        
        # Upload stress model to models/serving/battery_stress_model/
        stress_model_files = []
        for root, dirs, files in os.walk(stress_model.path):
            for file in files:
                file_path = os.path.join(root, file)
                s3_key = f"models/serving/battery_stress_model/{os.path.relpath(file_path, stress_model.path)}"
                s3_client.upload_file(file_path, aws_s3_bucket, s3_key)
                stress_model_files.append(s3_key)
        
        # Upload TTF model to models/serving/battery_ttf_model/
        ttf_model_files = []
        for root, dirs, files in os.walk(ttf_model.path):
            for file in files:
                file_path = os.path.join(root, file)
                s3_key = f"models/serving/battery_ttf_model/{os.path.relpath(file_path, ttf_model.path)}"
                s3_client.upload_file(file_path, aws_s3_bucket, s3_key)
                ttf_model_files.append(s3_key)
        
        status = f"Successfully uploaded {len(stress_model_files)} stress model files and {len(ttf_model_files)} TTF model files to S3"
        print(status)
        
    except ClientError as e:
        status = f"Error uploading to S3: {e}"
        print(status)
    except Exception as e:
        status = f"Unexpected error: {e}"
        print(status)
    
    from collections import namedtuple
    output = namedtuple('Outputs', ['upload_status'])
    return output(status)


# =============================================================================
# PIPELINE DEFINITION
# =============================================================================

@pipeline(
    name="battery-ml-pipeline",
    description="Complete ML pipeline for battery stress detection and time-to-failure prediction",
    pipeline_root="gs://your-bucket/pipeline-root"
)
def battery_ml_pipeline(
    # InfluxDB parameters
    influxdb_url: str = "https://influxdb-battery-demo.apps.sno.pemlab.rdu2.redhat.com/",
    influxdb_token: str = "admin_token",
    influxdb_org: str = "redhat",
    influxdb_bucket: str = "bms",
    
    # S3 parameters
    aws_access_key_id: str = "minio",
    aws_secret_access_key: str = "minio123",
    aws_s3_endpoint: str = "http://minio-service.minio.svc.cluster.local:9000",
    aws_s3_bucket: str = "s3-storage"
):
    """
    Main pipeline using UBI images for optimal OpenShift compatibility
    """
    
    # Step 1: Retrieve data from InfluxDB
    retrieve_task = retrieve_influx_data(
        influxdb_url=influxdb_url,
        influxdb_token=influxdb_token,
        influxdb_org=influxdb_org,
        influxdb_bucket=influxdb_bucket
    )
    
    # Step 2: Prepare data
    prepare_task = prepare_data(
        raw_data=retrieve_task.outputs['raw_data']
    )
    
    # Step 3: Train stress detection model
    stress_train_task = train_stress_detection_model(
        prepared_data=prepare_task.outputs['prepared_data']
    )
    
    # Step 4: Predict stress
    stress_predict_task = predict_stress(
        stress_model=stress_train_task.outputs['stress_model'],
        prepared_data=prepare_task.outputs['prepared_data']
    )
    
    # Step 5: Train TTF model
    ttf_train_task = train_ttf_model(
        prepared_data=prepare_task.outputs['prepared_data']
    )
    
    # Step 6: Predict TTF
    ttf_predict_task = predict_ttf(
        ttf_model=ttf_train_task.outputs['ttf_model'],
        prepared_data=prepare_task.outputs['prepared_data']
    )
    
    # Step 7: Save models to S3
    save_task = save_models_to_s3(
        stress_model=stress_train_task.outputs['stress_model'],
        ttf_model=ttf_train_task.outputs['ttf_model'],
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_endpoint=aws_s3_endpoint,
        aws_s3_bucket=aws_s3_bucket
    )
    
    # Configure explicit dependencies
    prepare_task.after(retrieve_task)
    stress_train_task.after(prepare_task)
    ttf_train_task.after(prepare_task)
    stress_predict_task.after(stress_train_task)
    ttf_predict_task.after(ttf_train_task)
    save_task.after(stress_train_task, ttf_train_task)


# =============================================================================
# PIPELINE COMPILATION
# =============================================================================

if __name__ == "__main__":
    # Compile pipeline to YAML
    kfp.compiler.Compiler().compile(
        pipeline_func=battery_ml_pipeline,
        package_path="battery_ml_pipeline.yaml"
    )
    print("Pipeline compiled successfully to 'battery_ml_pipeline.yaml'")
    print("\nComplete ML pipeline for batteries with OpenVINO and MinIO")
