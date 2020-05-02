install.packages("caret") 
install.packages("keras")
install.packages("tensorflow")

install.packages('R.utils')

library(keras) # load keras
library(tensorflow)
install_tensorflow() # build a tensorflow environment
#install_keras()

library(caret)
library(pROC)
library(ggplot2)
library(rpart)
library(MASS)
library(Matrix)
library(tidyverse) # metapackage with lots of helpful functions
library(data.table)
library(ggthemes)


  if(!file.exists("zip.train.gz")){
    download.file("https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.train.gz","zip.train.gz")
  }

zip.train <- data.table::fread("zip.train.gz")


zip <- as.data.frame(zip.train)

zip.train.80 <- sample(nrow(zip.train),0.8*nrow(zip.train))##随机无回放抽取百




batch_size <- 128
num_classes <- 10
epochs <- 12

# Input image dimensions
img_rows <- 16
img_cols <- 16

input_shape <- c(img_rows, img_cols, 1)



train_set <- data.matrix(zip[zip.train.80,])
target <- data.matrix(unlist(train_set[1:nrow(train_set),1]))
train_set  <- train_set[,-1]
fold_vec  = 5
set.seed(1234)
folds = createFolds(factor(target), k = 5, list = FALSE)

test_acc <- cbind.data.frame(c(1:5),rep(0,5),rep(0,5),rep(0,5)) 
test_loss <- cbind.data.frame(c(1:5),rep(0,5),rep(0,5),rep(0,5)) 
colnames(test_acc) <- c('fold_vec','accuracy_con','accuracy_den','baseline')
colnames(test_loss) <- c('fold_vec','accuracy_con','accuracy_den','baseline')
con_epochs  <- rep(0,10)
den_epochs  <- rep(0,10)

for (i in 1:10){
  

for (this.round in 1:fold_vec){      
  valid <- c(1:length(target)) [folds == this.round]
  dev <- c(1:length(target)) [folds != this.round]
  
  x.train<-  data.matrix(train_set[dev,])
  
  x.train <- array(
    unlist(x.train[1:nrow(x.train),]),
    c(nrow(x.train), 16, 16, 1))
  
  y.train <- data.matrix(train_set[valid,])
  y.train <- array(
    unlist(y.train[1:nrow(y.train),]),
    c(nrow(y.train), 16, 16, 1))
  
  
  x.test <- target[dev]
  y.test <- target[valid]
  ###
  
  
  #CONVOLUTIONAL
  
  model <- keras_model_sequential() %>%
    layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                  input_shape = dim(x.train)[-1]) %>% 
    layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
    layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
    layer_dropout(rate = 0.25) %>% 
    layer_flatten() %>% 
    layer_dense(units = 128, activation = 'relu') %>% 
    layer_dropout(rate = 0.5) %>% 
    layer_dense(units = num_classes, activation = 'softmax')
  model%>% keras::compile(
    loss = 'sparse_categorical_crossentropy', 
    optimizer = 'adam', 
    metrics = c('accuracy') 
  )
 
 
  ## deep dense model.
  model2 <- keras_model_sequential() %>%
    layer_flatten(input_shape = dim(x.train)[-1]) %>% 
    layer_dense(units = 270, activation = 'relu') %>% 
    layer_dense(units = 270, activation = 'relu') %>% 
    layer_dense(units = 128, activation = 'relu') %>% 
    layer_dense(units = num_classes, activation = 'softmax')

  model2%>% keras::compile(
    loss = 'sparse_categorical_crossentropy', 
    optimizer = 'adam', 
    metrics = c('accuracy') 
  )
  
  
  
  model %>% fit(x.train, x.test,epochs = 19+i, batch_size = 128,validation_split = 0.2) 
  
  loss_and_metrics<- model %>% keras::evaluate(y.train, y.test, batch_size = 128)
  
  
  model2 %>% fit(x.train, x.test,epochs = 19+i, batch_size = 128,validation_split = 0.2) 
  
  loss_and_metrics2<- model2 %>% keras::evaluate(y.train, y.test, batch_size = 128)
  
  test_acc[this.round,2] <- loss_and_metrics$accuracy
  test_acc[this.round,3] <- loss_and_metrics2$accuracy
  test_acc[this.round,4] <- length(y.test[y.test==0])/length(y.test)
  test_loss[this.round,2] <- loss_and_metrics$loss
  test_loss[this.round,3] <- loss_and_metrics2$loss
  test_loss[this.round,4] <- length(y.test[y.test==0])/length(y.test)
}
  con_epochs[i]  <- mean(test_acc[,2])
  den_epochs[i] <- mean(test_acc[,3])
}

best_epochs_con <-  which.min(con_epochs)+19

best_epochs_den <-  which.min(den_epochs)+19


####fit the best model
#CONVOLUTIONAL
dtrain<- data.matrix(train_set)

dtrain <- array(
  unlist(dtrain[1:nrow(dtrain),]),
  c(nrow(dtrain), 16, 16, 1))

test_set <- data.matrix(zip[-zip.train.80,])##
target_y <- data.matrix(unlist(test_set[1:nrow(test_set),1]))

test_set  <- data.matrix(test_set[,-1])

dtest <- array(
  unlist(test_set[1:nrow(test_set),]),
  c(nrow(test_set), 16, 16, 1))


model3 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = dim(dtrain)[-1]) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes, activation = 'softmax')
model3%>% keras::compile(
  loss = 'sparse_categorical_crossentropy', 
  optimizer = 'adam', 
  metrics = c('accuracy') 
)


## deep dense model.
model4 <- keras_model_sequential() %>%
  layer_flatten(input_shape = dtrain) %>% 
  layer_dense(units = 270, activation = 'relu') %>% 
  layer_dense(units = 270, activation = 'relu') %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dense(units = num_classes, activation = 'softmax')

model4%>% keras::compile(
  loss = 'sparse_categorical_crossentropy', 
  optimizer = 'adam', 
  metrics = c('accuracy') 
)



model3 %>% fit(dtrain, target,epochs = best_epochs_con, batch_size = 128) 

loss_and_metrics3<- model3 %>% keras::evaluate(dtest, target_y, batch_size = 128)
loss_and_metrics3$accuracy

model4 %>% fit(dtrain, target,epochs = best_epochs_den, batch_size = 128) 

loss_and_metrics4<- model4 %>% keras::evaluate(dtest, target_y, batch_size = 128)
loss_and_metrics4$accuracy


##########dotplot


data_melt<-melt (test_acc[,-1])

ggplot(data_melt, aes(x =  value, y = variable, color=variable, fill= variable)) + 
  geom_dotplot(binaxis='y', stackdir='center', binwidth = 0.05)+
  theme(panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.border = element_rect(colour="black",fill=NA))