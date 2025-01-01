##### Libraries

library(dplyr)
library(keras)
library(tensorflow)


##### Data Prepreparation & Pipeline

# Assuming sdir points to the directory with images and their labels are inferred from directory structure or a file
sdir <- "C:/Users/Nicho/Desktop/C-NMC_Leukemia/training_data"

# List files and create a dataframe with file paths and labels
file_paths <- list.files(sdir, full.names = TRUE, recursive = TRUE)
labels <- sapply(strsplit(basename(dirname(file_paths)), "/"), `[`, 1) # This might need adjustment based on your directory structure

data_df <- data.frame(filepaths = file_paths, labels = as.factor(labels))

# Splitting data into training, validation, and test sets
set.seed(123) # For reproducibility
# Calculating sizes for each dataset
n <- nrow(data_df)
n_train <- round(n * trsplit)
n_valid <- round(n * vsplit)

# Shuffle data
data_df <- data_df[sample(nrow(data_df)), ]

# Creating splits
train_df <- data_df[1:n_train, ]
temp_df <- data_df[(n_train + 1):n, ]
valid_df <- temp_df[1:n_valid, ]
test_df <- temp_df[(n_valid + 1):nrow(temp_df), ]


# Calculating test batch size
length <- nrow(test_df)
test_batch_size <- max(Filter(function(n) length %% n == 0 && length / n <= 80, 1:length))
test_steps <- length %/% test_batch_size

# Preprocessing function
scalar <- function(img) {
  img # EfficientNet expects pixels in range 0 to 255 so there is no need for scaling
}

# Image Data Generators
datagen <- image_data_generator(preprocessing_function = scalar)
train_gen <- flow_from_dataframe(datagen, dataframe = train_df, x_col = "filepaths",
                                 y_col = "labels", target_size = img_size,
                                 class_mode = "categorical", color_mode = "rgb",
                                 shuffle = TRUE, batch_size = batch_size)
test_gen <- flow_from_dataframe(datagen, dataframe = test_df, x_col = "filepaths",
                                y_col = "labels", target_size = img_size,
                                class_mode = "categorical", color_mode = "rgb",
                                shuffle = TRUE, batch_size = test_batch_size)
valid_gen <- flow_from_dataframe(datagen, dataframe = valid_df, x_col = "filepaths",
                                 y_col = "labels", target_size = img_size,
                                 class_mode = "categorical", color_mode = "rgb",
                                 shuffle = TRUE, batch_size = batch_size)

# Getting class names and count
classes <- train_gen$class_indices
class_count <- length(classes)
train_steps <- ceiling(nrow(train_df) / batch_size)



##### Modeling


model_name <- "EfficientNetB3"
base_model <- application_efficientnet_b3(include_top = FALSE, weights = "imagenet",
                                          input_shape = c(img_size[1], img_size[2], channels), pooling = "max")
x <- base_model$output
x <- layer_batch_normalization(axis = -1, momentum = 0.99, epsilon = 0.001)(x)
x <- layer_dense(x, units = 256, kernel_regularizer = regularizer_l2(0.016),
                 activity_regularizer = regularizer_l1(0.006),
                 bias_regularizer = regularizer_l1(0.006), activation = "relu")
x <- layer_dropout(x, rate = 0.45, seed = 123)
output <- layer_dense(x, units = class_count, activation = "softmax")
model <- keras_model(inputs = base_model$input, outputs = output)
model %>% compile(optimizer = optimizer_adamax(lr = 0.001),
                  loss = "categorical_crossentropy", metrics = "accuracy")



##### Training


# Callbacks setup
LRA_callback <- function(...) {
  # Define or adapt a callback for learning rate adjustment in R
}

# Assuming LRA() equivalent setup in R
callbacks <- list(LRA_callback())

epochs <- 40

# Assuming `model%>%fit()` supports similar parameters as in Python
history <- model %>% fit_generator(train_gen,
                                   epochs = epochs,
                                   verbose = 0,
                                   callbacks = callbacks,
                                   validation_data = valid_gen,
                                   validation_steps = NULL,
                                   shuffle = FALSE)



##### Evaluation


preds <- predict_generator(model, test_gen)
print(preds)




















































