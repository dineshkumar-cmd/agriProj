"""
Stage 1 - train the lightweight single-class ("potato") localizer.

Any Ultralytics YOLO variant works here; default is the smallest (v8n)
since this stage only needs to answer "is this a potato", not grade it.
"""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt",
                         help="yolov8n.pt, yolov8s.pt, yolo11n.pt, etc.")
    parser.add_argument("--data", default="data/localizer/data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--fraction", type=float, default=1.0, 
                        help="Fraction of dataset to use for training (0 to 1.0).")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        patience=5,          # early stopping, matches base paper's approach
        project="runs/localizer",
        name="potato_localizer",
    )

    # Built-in validation gives mAP@0.5 and mAP@0.5:0.95 directly.
    metrics = model.val()
    print(metrics)


if __name__ == "__main__":
    main()
