import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from config import Config
from data import class_weights, prepare_generators
from evaluate import evaluate_model
from models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--model", default="dnbcd",
                        choices=["dnbcd", "densenet", "mobilenet", "resnet50", "vgg19"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    config = Config(model_name=args.model)
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.data:
        config.dataset_dir = args.data
    config.work_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(config.work_dir / "training.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    random.seed(config.seed)
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    logger.info("Dataset: %s", config.dataset_dir)
    logger.info("Model: %s | epochs=%d | batch_size=%d", config.model_name, config.epochs, config.batch_size)
    logger.info("Visible GPUs: %s", tf.config.list_physical_devices("GPU"))
    train_gen, validation_gen, test_gen = prepare_generators(config)
    weights = class_weights(train_gen)
    logger.info("Class indices: %s", train_gen.class_indices)
    logger.info("Class weights: %s", weights)

    model = build_model(config.model_name, config.image_size, fine_tune=False)
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        ModelCheckpoint(config.checkpoint_path, monitor="val_accuracy", save_best_only=True),
        CSVLogger(config.work_dir / "training.csv"),
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ]
    model.fit(train_gen, epochs=config.epochs, validation_data=validation_gen,
              class_weight=weights, callbacks=callbacks)

    if config.model_name == "dnbcd":
        # The original DNBCD notebook unfreezes DenseNet for a second stage.
        for layer in model.layers:
            layer.trainable = True
        model.compile(optimizer=Adam(learning_rate=1e-4),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_gen, epochs=config.epochs, validation_data=validation_gen,
                  class_weight=weights, callbacks=callbacks)

    model.save(config.work_dir / f"{config.model_name}_final.keras")
    metrics = evaluate_model(model, test_gen, config.work_dir)
    (config.work_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved checkpoint: %s", config.checkpoint_path)
    logger.info("Final metrics: %s", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
