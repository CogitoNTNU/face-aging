
def generator(latent_dim=122,y_dim=6):

    input_latent = Input((latent_dim,))

    input_label= Input((y_dim,))

    gen = Concatenate()([input_latent, input_label])
    gen = Dense(1024*4*4)(gen)
    gen = ReLU()(gen)
    gen = Reshape((4,4,1024))(gen)

    gen = Conv2DTranspose(512,4,strides=2,padding="same")(gen) #8,8
    gen = ReLU()(gen)

    gen = Conv2DTranspose(256,4,strides=2,padding="same")(gen) #8,8
    #gen = BatchNormalization()(gen)
    gen = ReLU()(gen)

    gen = Conv2DTranspose(128,4,strides=2,padding="same")(gen) #8,8
    gen = ReLU()(gen)

    gen = Conv2DTranspose(CHANNELS,4,strides=2,padding="same")(gen) #8,8
    gen = Activation("tanh")(gen)

    gen = Model([input_latent,input_label],gen)

    #plot_model(gen, to_file="model.png")
    return gen


def discriminator(height,width,channels,label_dim,learning_rate,embedding_dim=64):

    input_image = Input((height,width,channels))
    input_label = Input((label_dim,))

    label = Dense(HEIGHT * WIDTH)(input_label)
    label = LeakyReLU()(label)
    label = Reshape((HEIGHT, WIDTH, 1))(label)

    disc = Concatenate()([input_image,label])


    disc = Conv2D(128, 4,strides=2,padding="same")(disc)
    disc = LeakyReLU()(disc)

    disc = Conv2D(256, 4,strides=2,padding="same")(disc)
    disc = LeakyReLU()(disc)

    disc = Conv2D(512, 4,strides=2,padding="same")(disc)
    #disc = BatchNormalization()(disc)
    disc = LeakyReLU()(disc)

    disc = Conv2D(1024, 4,strides=2,padding="same")(disc)
    disc = LeakyReLU()(disc)

    disc = Flatten()(disc)

    disc = Dropout(0.4)(disc)

    disc = Dense(1, activation="sigmoid")(disc)

    disc = Model([input_image,input_label],disc)

    discriminator_optimizer = Adam(lr=LEARNING_RATE,beta_1=0.5)

    disc.compile(discriminator_optimizer, loss="binary_crossentropy",metrics=['binary_accuracy'])

    return disc
