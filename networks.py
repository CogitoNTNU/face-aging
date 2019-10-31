from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Concatenate, Input, BatchNormalization, Reshape, Conv2DTranspose, Dropout
from tensorflow.keras.layers import ReLU, Activation, LeakyReLU


def build_generator(latent_dim=121, age_dim=6, gender_dim=1, output_channels=3):
    
    # Input parameters
    input_latent = Input((latent_dim,))
    input_label = Input((age_dim,))
    input_gender = Input((gender_dim,))
    
    # Concat
    gen = Concatenate()([input_latent, input_label, input_gender])
    
    # Expand dimensions
    gen = Dense(1024*4*4)(gen)
    gen = ReLU()(gen)
    
    # Reshape feature map
    gen = Reshape((4,4,1024))(gen)
    
    # Upsample image
    gen = Conv2DTranspose(512,4,strides=2,padding="same")(gen)
    gen = ReLU()(gen)

    # Upsample image
    gen = Conv2DTranspose(256,4,strides=2,padding="same")(gen)
    gen = ReLU()(gen)

    # Upsample image
    gen = Conv2DTranspose(128,4,strides=2,padding="same")(gen)
    gen = ReLU()(gen)

    # Upsample image
    gen = Conv2DTranspose(output_channels,4,strides=2,padding="same")(gen)
    gen = Activation("tanh")(gen)

    # Create model
    gen = Model([input_latent,input_label,input_gender], gen)

    return gen


def build_discriminator(image_shape=(64, 64, 3), age_dim=6, gender_dim=1):

    # Create input
    input_image = Input(image_shape)
    input_label = Input((age_dim, ))
    input_gen = Input((gender_dim, ))

    # Concatinate target label as an image channel
    label = Concatenate()([input_label, input_gen])
    label = Dense(image_shape[0] * image_shape[1])(label)
    label = LeakyReLU()(label)
    label = Reshape((image_shape[0], image_shape[1], 1))(label)
    disc = Concatenate()([input_image,label])

    # TODO: change to normal conv params
    disc = Conv2D(128, 4, strides=2,padding="same")(disc)
    disc = LeakyReLU()(disc)

    disc = Conv2D(256, 4, strides=2,padding="same")(disc)
    disc = LeakyReLU()(disc)

    disc = Conv2D(512, 4, strides=2, padding="same")(disc)
    disc = LeakyReLU()(disc)

    disc = Conv2D(1024, 4, strides=2, padding="same")(disc)
    disc = LeakyReLU()(disc)

    # Flatten
    disc = Flatten()(disc)
    disc = Dropout(0.4)(disc)
    
    disc = Dense(2048)(disc)
    disc = LeakyReLU()(disc)
    # Final prediction
    disc = Dense(1, activation="sigmoid")(disc)

    # Create model
    disc = Model([input_image,input_label,input_gen], disc)

    return disc

def build_encoder(image_shape=(64, 64, 3), latent_dim=128):
    """
    Encoder Network
    """
    input_layer = Input(shape=image_shape)

    # 1st Convolutional Block
    enc = Conv2D(filters=32, kernel_size=5, strides=2, padding='same')(input_layer)
    # enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # 2nd Convolutional Block
    enc = Conv2D(filters=64, kernel_size=5, strides=2, padding='same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # 3rd Convolutional Block
    enc = Conv2D(filters=128, kernel_size=5, strides=2, padding='same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # 4th Convolutional Block
    enc = Conv2D(filters=256, kernel_size=5, strides=2, padding='same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # Flatten layer
    enc = Flatten()(enc)

    # 1st Fully Connected Layer
    enc = Dense(4096)(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # Second Fully Connected Layer
    enc = Dense(latent_dim)(enc)

    # Create a model
    model = Model(inputs=[input_layer], outputs=[enc])
    return model

if __name__ == "__main__":
    encoder = build_encoder()
    gen = build_generator()
    disc = build_discriminator()
