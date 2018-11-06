library(ggplot2)
library(cowplot)
library(gridExtra)

df_pretrained_SGD <- read.table("ResNet50_PreTrained_SGD_KFold_0.log", 
                                header = TRUE,
                                sep = ",")

df_pretrained_RMSprop <- read.table("ResNet50_PreTrained_RMSprop_KFold_0.log", 
                                header = TRUE,
                                sep = ",")

df_pretrained_Adagrad <- read.table("ResNet50_PreTrained_Adagrad_KFold_0.log", 
                                header = TRUE,
                                sep = ",")

df_pretrained_Adadelta <- read.table("../../logs/ResNet50/ResNet50_PreTrained_Adadelta_KFold_0.log", 
                                header = TRUE,
                                sep = ",")

df_pretrained_Adam <- read.table("ResNet50_PreTrained_Adam_KFold_0.log", 
                                header = TRUE,
                                sep = ",")

df_pretrained_Adamax <- read.table("ResNet50_PreTrained_Adamax_KFold_0.log", 
                                header = TRUE,
                                sep = ",")

df_pretrained_Adam_AMSGrad <- read.table("ResNet50_PreTrained_Adam_AMSGrad_KFold_0.log", 
                                   header = TRUE,
                                   sep = ",")

df_pretrained_Adam_AMSGrad_LR_0.0001 <- read.table("ResNet50_PreTrained_Adam_AMSGrad_LR_0.0001_KFold_0.log", 
                                   header = TRUE,
                                   sep = ",")


df_pretrained_SGD$optimizer <- c(rep("SGD", nrow(df_pretrained_SGD)))
df_pretrained_RMSprop$optimizer <- c(rep("RMSprop", nrow(df_pretrained_RMSprop)))
df_pretrained_Adagrad$optimizer <- c(rep("Adagrad", nrow(df_pretrained_Adagrad)))
df_pretrained_Adadelta$optimizer <- c(rep("Adadelta", nrow(df_pretrained_Adadelta)))
df_pretrained_Adam$optimizer <- c(rep("Adam", nrow(df_pretrained_Adam)))
df_pretrained_Adamax$optimizer <- c(rep("Adamax", nrow(df_pretrained_Adamax)))
df_pretrained_Adam_AMSGrad$optimizer <- c(rep("Adam_AMSGrad", nrow(df_pretrained_Adam_AMSGrad)))
df_pretrained_Adam_AMSGrad_LR_0.0001$optimizer <- c(rep("Adam_AMSGrad_LR_0.0001", nrow(df_pretrained_Adam_AMSGrad_LR_0.0001)))

df <- rbind.data.frame(df_pretrained_SGD, 
                       df_pretrained_RMSprop, 
                       df_pretrained_Adagrad,
                       df_pretrained_Adadelta,
                       df_pretrained_Adam,
                       df_pretrained_Adamax,
                       df_pretrained_Adam_AMSGrad,
                       df_pretrained_Adam_AMSGrad_LR_0.0001)

plot_acc <- ggplot(data = df, aes(x = epoch, y = acc, color = optimizer)) + 
  geom_line() + xlab("Epochs") + ylab("Training accuracy") + 
  theme(legend.title = element_blank(),
        legend.position = c(0.6, 0.2),
        legend.text = element_text(size = 10))

plot_loss <- ggplot(data = df, aes(x = epoch, y = loss, color = optimizer)) + 
  geom_line() + xlab("Epochs") + ylab("Training loss") + 
  theme(legend.title = element_blank(),
        legend.position = c(0.6, 0.8),
        legend.text = element_text(size = 10))

plot_val_acc <- ggplot(data = df, aes(x = epoch, y = val_acc, color = optimizer)) + 
  geom_line() + xlab("Epochs") + ylab("Validation accuracy") + 
  theme(legend.title = element_blank(),
        legend.position = c(0.6, 0.2),
        legend.text = element_text(size = 10))

plot_val_loss <- ggplot(data = df, aes(x = epoch, y = val_loss, color = optimizer)) + 
  geom_line() + xlab("Epochs") + ylab("Validation loss") + 
  theme(legend.title = element_blank(),
        legend.position = c(0.6, 0.8),
        legend.text = element_text(size = 10))

p <- plot_grid(plot_acc, plot_loss, plot_val_acc, plot_val_loss, nrow = 2, ncol = 2)
png("pre_trained.png", height = 12, width = 12, units = "in", res = 300)
print(p)
dev.off()


df_fromscratch_SGD <- read.table("ResNet50_FromScratch_SGD_KFold_0.log", 
                                header = TRUE,
                                sep = ",")

df_fromscratch_RMSprop <- read.table("ResNet50_FromScratch_RMSprop_KFold_0.log", 
                                    header = TRUE,
                                    sep = ",")

df_fromscratch_Adagrad <- read.table("ResNet50_FromScratch_Adagrad_KFold_0.log", 
                                    header = TRUE,
                                    sep = ",")

df_fromscratch_Adadelta <- read.table("ResNet50_FromScratch_Adadelta_KFold_0.log", 
                                     header = TRUE,
                                     sep = ",")

df_fromscratch_Adam <- read.table("ResNet50_FromScratch_Adam_KFold_0.log", 
                                 header = TRUE,
                                 sep = ",")

df_fromscratch_Adamax <- read.table("ResNet50_FromScratch_Adamax_KFold_0.log", 
                                   header = TRUE,
                                   sep = ",")

df_fromscratch_Adam_AMSGrad <- read.table("ResNet50_FromScratch_Adam_AMSGrad_KFold_0.log", 
                                         header = TRUE,
                                         sep = ",")

df_fromscratch_Adam_AMSGrad_LR_0.0001 <- read.table("ResNet50_FromScratch_Adam_AMSGrad_LR_0.0001_KFold_0.log", 
                                                   header = TRUE,
                                                   sep = ",")

df_fromscratch_SGD <- df_fromscratch_SGD[0:40, ]
df_fromscratch_RMSprop <- df_fromscratch_RMSprop[0:40, ]
df_fromscratch_Adagrad <- df_fromscratch_Adagrad[0:40, ]
df_fromscratch_Adadelta <- df_fromscratch_Adadelta[0:40, ]
df_fromscratch_Adam <- df_fromscratch_Adam[0:40, ]
df_fromscratch_Adamax <- df_fromscratch_Adamax[0:40, ]
df_fromscratch_Adam_AMSGrad <- df_fromscratch_Adam_AMSGrad[0:40, ]
df_fromscratch_Adam_AMSGrad_LR_0.0001 <- df_fromscratch_Adam_AMSGrad_LR_0.0001[0:40, ]

df_fromscratch_SGD$epoch <- seq(0, 39)
df_fromscratch_RMSprop$epoch <- seq(0, 39)
df_fromscratch_Adagrad$epoch <- seq(0, 39)
df_fromscratch_Adadelta$epoch <- seq(0, 39)
df_fromscratch_Adam$epoch <- seq(0, 39)
df_fromscratch_Adamax$epoch <- seq(0, 39)
df_fromscratch_Adam_AMSGrad$epoch <- seq(0, 39)
df_fromscratch_Adam_AMSGrad_LR_0.0001$epoch <- seq(0, 39)

df_fromscratch_SGD$optimizer <- c(rep("SGD", nrow(df_fromscratch_SGD)))
df_fromscratch_RMSprop$optimizer <- c(rep("RMSprop", nrow(df_fromscratch_RMSprop)))
df_fromscratch_Adagrad$optimizer <- c(rep("Adagrad", nrow(df_fromscratch_Adagrad)))
df_fromscratch_Adadelta$optimizer <- c(rep("Adadelta", nrow(df_fromscratch_Adadelta)))
df_fromscratch_Adam$optimizer <- c(rep("Adam", nrow(df_fromscratch_Adam)))
df_fromscratch_Adamax$optimizer <- c(rep("Adamax", nrow(df_fromscratch_Adamax)))
df_fromscratch_Adam_AMSGrad$optimizer <- c(rep("Adam_AMSGrad", nrow(df_fromscratch_Adam_AMSGrad)))
df_fromscratch_Adam_AMSGrad_LR_0.0001$optimizer <- c(rep("Adam_AMSGrad_LR_0.0001", nrow(df_fromscratch_Adam_AMSGrad_LR_0.0001)))


df <- rbind.data.frame(df_fromscratch_SGD, 
                       df_fromscratch_RMSprop, 
                       df_fromscratch_Adagrad,
                       df_fromscratch_Adadelta,
                       df_fromscratch_Adam,
                       df_fromscratch_Adamax,
                       df_fromscratch_Adam_AMSGrad,
                       df_fromscratch_Adam_AMSGrad_LR_0.0001)

plot_acc <- ggplot(data = df, aes(x = epoch, y = acc, color = optimizer)) + 
  geom_line() + xlab("Epochs") + ylab("Training accuracy") + 
  theme(legend.title = element_blank(),
        legend.position = c(0.6, 0.2),
        legend.text = element_text(size = 10))

plot_loss <- ggplot(data = df, aes(x = epoch, y = loss, color = optimizer)) + 
  geom_line() + xlab("Epochs") + ylab("Training loss") + 
  theme(legend.title = element_blank(),
        legend.position = c(0.6, 0.8),
        legend.text = element_text(size = 10))

plot_val_acc <- ggplot(data = df, aes(x = epoch, y = val_acc, color = optimizer)) + 
  geom_line() + xlab("Epochs") + ylab("Validation accuracy") + 
  theme(legend.title = element_blank(),
        legend.position = c(0.6, 0.2),
        legend.text = element_text(size = 10))

plot_val_loss <- ggplot(data = df, aes(x = epoch, y = val_loss, color = optimizer)) + 
  geom_line() + xlab("Epochs") + ylab("Validation loss") + 
  theme(legend.title = element_blank(),
        legend.position = c(0.6, 0.8),
        legend.text = element_text(size = 10))

p <- plot_grid(plot_acc, plot_loss, plot_val_acc, plot_val_loss, nrow = 2, ncol = 2)
png("from_scratch.png", height = 12, width = 12, units = "in", res = 300)
print(p)
dev.off()

