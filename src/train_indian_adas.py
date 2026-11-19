"""
UNLET-ADAS: Indian ADAS & Unstructured Road Environment Training Pipeline
==========================================================================
Fine-tunes YOLOv8 for Indian traffic conditions, specifically supporting:
1. Indian Vehicles:
   - Auto-Rickshaws (Bajaj RE, TVS King, Piaggio Ape, e-rickshaws)
   - Commercial Indian Trucks (Tata Ace, Ashok Leyland, Eicher)
   - Indian Transit & Sleeper Buses
   - 2-Wheelers (Motorcycles, Scooters, Mopeds) & Bicycles
2. Roadway Animals & Hazards:
   - Stray Cattle / Cows / Water Buffaloes (frequent Indian road collision hazard)
   - Stray Dogs, Cats, Horses, Sheep / Goats, Birds
3. Indian Road Infrastructure:
   - Indian Traffic Signs (mandatory, cautionary, informative, speed limit, bump ahead)
   - Road Potholes & surface degradations

Datasets Supported:
- Roboflow "indian-vehicles-and-traffic" / "indian-traffic-signs1"
- India Driving Dataset (IDD) YOLO conversion
- Custom labeled Indian roadway images

Usage:
    pip install roboflow ultralytics
    # Train using a free Roboflow API key:
    python src/train_indian_adas.py --roboflow_key YOUR_KEY --epochs 60

    # Or point to an existing local dataset:
    python src/train_indian_adas.py --data_yaml /path/to/data.yaml --epochs 60
"""

import os
import sys
import shutil
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Full taxonomy for Indian Autonomous Driving & ADAS
INDIAN_ADAS_TAXONOMY = [
    'Person',          # 0
    'Bicycle',         # 1
    'Car',             # 2
    'Motorcycle',      # 3
    'Auto-Rickshaw',   # 4: 3-wheeler passenger / cargo / e-rickshaw
    'Bus',             # 5
    'Truck',           # 6: Commercial freight / mini-truck
    'Tractor',         # 7: Agricultural vehicle
    'Cattle / Cow',    # 8: Cows, bulls, buffaloes
    'Dog',             # 9: Stray dogs
    'Bird',            # 10
    'Traffic Light',   # 11
    'Stop Sign',       # 12
    'Road Sign',       # 13: Generic caution/speed sign
    'Pothole',         # 14
]


def create_default_data_yaml(dest_dir: str) -> str:
    """Creates a standardized data.yaml for Indian traffic detection."""
    yaml_path = os.path.join(dest_dir, 'indian_adas.yaml')
    train_path = os.path.join(dest_dir, 'images', 'train')
    val_path = os.path.join(dest_dir, 'images', 'val')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    content = f"""# UNLET-ADAS Indian Roadways Dataset Configuration
path: {os.path.abspath(dest_dir)}
train: images/train
val: images/val

names:
"""
    for i, name in enumerate(INDIAN_ADAS_TAXONOMY):
        content += f"  {i}: '{name}'\n"

    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Dataset config created at: {yaml_path}")
    return yaml_path


def download_indian_dataset(api_key: str, dest_dir: str,
                            workspace: str = 'indian-traffic',
                            project_slug: str = 'indian-vehicles-dataset',
                            version_num: int = 1) -> str:
    """Downloads curated Indian vehicles & traffic signs dataset from Roboflow."""
    from roboflow import Roboflow
    print(f"Connecting to Roboflow: {workspace}/{project_slug} v{version_num}...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_slug)
    dataset = project.version(version_num).download('yolov8', location=dest_dir)
    return os.path.join(dataset.location, 'data.yaml')


def train(data_yaml: str,
          model_name: str = 'yolov8s.pt',
          epochs: int = 60,
          batch_size: int = 16,
          imgsz: int = 640,
          save_dir: str = 'app',
          device: str = ''):
    """Fine-tunes YOLOv8 on the Indian traffic dataset."""
    from ultralytics import YOLO

    print(f"\n=======================================================")
    print(f"Starting UNLET-ADAS Indian Traffic Model Training")
    print(f"Base model : {model_name}")
    print(f"Config     : {data_yaml}")
    print(f"Epochs     : {epochs} | Batch: {batch_size} | Size: {imgsz}")
    print(f"=======================================================\n")

    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device if device else None,
        plots=True,
        save=True,
        verbose=True,
    )

    best_pt = os.path.join(model.trainer.save_dir, 'weights', 'best.pt')
    if os.path.exists(best_pt):
        out_pt = os.path.join(save_dir, 'yolov8_indian_adas.pt')
        os.makedirs(save_dir, exist_ok=True)
        shutil.copy2(best_pt, out_pt)
        print(f"\nSuccess! Exported Indian ADAS model to: {out_pt}")
    return results


def main():
    parser = argparse.ArgumentParser(description="UNLET-ADAS Indian Traffic Training Pipeline")
    parser.add_argument('--roboflow_key', type=str, default='', help='Roboflow API key')
    parser.add_argument('--data_yaml', type=str, default='', help='Path to existing data.yaml')
    parser.add_argument('--epochs', type=int, default=60, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image resolution')
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='Base YOLOv8 model')
    parser.add_argument('--dest_dir', type=str, default='./data/indian_adas', help='Dataset directory')
    args = parser.parse_args()

    if args.data_yaml and os.path.exists(args.data_yaml):
        data_yaml = args.data_yaml
    elif args.roboflow_key:
        data_yaml = download_indian_dataset(args.roboflow_key, args.dest_dir)
    else:
        print("No API key or data_yaml provided. Generating template dataset structure...")
        data_yaml = create_default_data_yaml(args.dest_dir)
        print(f"\nTemplate ready! Populate '{args.dest_dir}/images/train' and run with --data_yaml {data_yaml}")
        return

    train(data_yaml=data_yaml, model_name=args.model, epochs=args.epochs,
          batch_size=args.batch_size, imgsz=args.imgsz)


if __name__ == '__main__':
    main()
