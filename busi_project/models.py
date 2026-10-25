from tensorflow.keras.applications import DenseNet121, MobileNet, ResNet50, VGG19
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def build_model(name: str, image_size: int = 128, fine_tune: bool = False):
    builders = {"dnbcd": DenseNet121, "densenet": DenseNet121,
                "mobilenet": MobileNet, "resnet50": ResNet50, "vgg19": VGG19}
    key = name.lower()
    if key not in builders:
        raise ValueError(f"Unknown model {name}; choose from {sorted(builders)}")
    base = builders[key](weights="imagenet", include_top=False,
                         input_shape=(image_size, image_size, 3))
    base.trainable = fine_tune
    x = GlobalAveragePooling2D()(base.output)
    if key == "dnbcd":
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.5)(x)
    output = Dense(3, activation="softmax")(x)
    return Model(base.input, output, name=name)
