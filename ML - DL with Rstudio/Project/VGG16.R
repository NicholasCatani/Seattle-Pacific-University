##### Libraries
library(keras)
library(tensorflow)
library(tidyverse)
library(caret)
library(reticulate)
py_install("h5py")
library(tensorflow)


##### Settings
create_df <- function(dataset) {
  image_paths <- list.files(dataset, full.names = TRUE, recursive = TRUE)
  labels <- ifelse(grepl("all$", dirname(image_paths)), "all", "hem")
  df <- tibble("Image Path" = image_paths, Label = labels)
  return(df)
}

train_dir <- "C:\\Users\\Nicho\\Desktop\\C-NMC_Leukemia\\training_data"
df <- create_df(train_dir)

set.seed(31)
train_index <- createDataPartition(df$Label, p = 0.7, list = FALSE)
train_df <- df[train_index, ]
remaining_df <- df[-train_index, ]

valid_index <- createDataPartition(remaining_df$Label, p = 0.5, list = FALSE)
valid_df <- remaining_df[valid_index, ]
test_df <- remaining_df[-valid_index, ]



##### Modeling
img_shape <- c(224, 224, 3)
VGG16_base_model <- application_vgg16(weights = "imagenet", include_top = FALSE, input_shape = img_shape)
VGG16_base_model$trainable <- FALSE

last_layer <- get_layer(VGG16_base_model, "block5_pool")
last_output <- last_layer$output
x <- layer_global_average_pooling_2d()(last_output)
x <- layer_dropout(rate = 0.5)(x)
predictions <- layer_dense(units = 2, activation = "sigmoid")(x)

VGG16_model <- keras_model(inputs = VGG16_base_model$input, outputs = predictions)
VGG16_model %>% compile(
  optimizer = optimizer_adamax(lr = 0.001),
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)



##### Pipeline
train_datagen <- image_data_generator(rescale = 1/255)
valid_datagen <- image_data_generator(rescale = 1/255)

train_gen <- flow_images_from_dataframe(
  dataframe = train_df,
  x_col = "Image Path",
  y_col = "Label",
  generator = train_datagen,
  target_size = c(224, 224),
  class_mode = "categorical",
  batch_size = 32
)

valid_gen <- flow_images_from_dataframe(
  dataframe = valid_df,
  x_col = "Image Path",
  y_col = "Label",
  generator = valid_datagen,
  target_size = c(224, 224),
  class_mode = "categorical",
  batch_size = 32
)



##### Training the Model
history <- fit_generator(
  VGG16_model,
  train_gen,
  steps_per_epoch = as.integer(train_gen$n / train_gen$batch_size),
  epochs = 20,
  validation_data = valid_gen,
  validation_steps = as.integer(valid_gen$n / valid_gen$batch_size)
)





























