import argparse
import logging
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import json
import numpy as np


from .config import IMAGE_SIZE, BATCH_SIZE, EPOCHS, MODEL_PATH

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logging.info(f"GPUs available: {len(gpus)}")
        for gpu in gpus:
            logging.info(f"GPU: {gpu}")
    else:
        logging.info("No GPU available, using CPU")

def load_data(dataset_path):
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2,
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )
    validation_datagen = ImageDataGenerator(
        validation_split=0.2,
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['mask', 'without_mask'],
        subset='training'
    )
    validation_generator = validation_datagen.flow_from_directory(
        dataset_path,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['mask', 'without_mask'],
        subset='validation',
        shuffle=False  # IMPORTANT
    )

    return train_generator, validation_generator

def build_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    
    # Freeze all layers initially
    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    
    # Compile with lower learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                  loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model, base_model

def fine_tune_model(model, base_model):
    # Unfreeze the last 20 layers for fine-tuning
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                  loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

def plot_roc_curve(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()

def main(args):
    setup_logging()
    check_gpu()

    logging.info("Loading data...")
    train_generator, validation_generator = load_data(args.dataset_path)

    logging.info(f"Class indices: {train_generator.class_indices}")

    # Save class_indices
    os.makedirs('models', exist_ok=True)
    with open('models/class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)

    logging.info("Building model...")
    model, base_model = build_model()
    model.summary()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ]

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
    class_weights = dict(enumerate(class_weights))
    logging.info(f"Class weights: {class_weights}")

    # Initial training with frozen base
    logging.info("Training initial model...")
    history1 = model.fit(
        train_generator,
        epochs=10,  # Initial epochs
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weights
    )

    # Fine-tune the model
    model = fine_tune_model(model, base_model)
    
    logging.info("Fine-tuning model...")
    history2 = model.fit(
        train_generator,
        epochs=EPOCHS,  # Remaining epochs for fine-tuning
        initial_epoch=10,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weights
    )

    logging.info("Evaluating model...")
    val_loss, val_accuracy, val_precision, val_recall = model.evaluate(validation_generator)
    logging.info(f"Validation Loss: {val_loss:.4f}")
    logging.info(f"Validation Accuracy: {val_accuracy:.4f}")
    logging.info(f"Validation Precision: {val_precision:.4f}")
    logging.info(f"Validation Recall: {val_recall:.4f}")

    # Get predictions and true labels for validation set
    validation_generator.reset()

    y_pred_prob = model.predict(validation_generator, verbose=1)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = validation_generator.classes

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Mask', 'No Mask']))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Threshold calibration using ROC
    from sklearn.metrics import roc_curve
    y_val_prob = model.predict(validation_generator)
    y_val_true = validation_generator.classes
    fpr, tpr, thresholds = roc_curve(y_val_true, y_val_prob)
    optimal_idx = np.argmax(tpr - fpr)  # Maximize TPR - FPR
    optimal_threshold = thresholds[optimal_idx]
    logging.info(f"Optimal threshold: {optimal_threshold:.2f}")

    # Save optimal threshold
    os.makedirs('models', exist_ok=True)
    with open('models/optimal_threshold.txt', 'w') as f:
        f.write(str(optimal_threshold))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Face Mask Detection Model")
    parser.add_argument('--dataset_path', type=str, default='../data', help='Path to data directory')
    args = parser.parse_args()
    main(args)
