# Plantilla de preprocesado de datos
dataset = read.csv('../Data.csv')
# # Tratamiento de los valores NA
# dataset$Age = ifelse(is.na(dataset$Age),
#                     ave(dataset$Age, FUN = function(x mean(x, na.rm = TRUE))),
#                      dataset$Age)
# dataset$Salary = ifelse(is.na(dataset$Salary),
#                     ave(dataset$Salary, FUN = function(x mean(x, na.rm = TRUE))),
#                     dataset$Salary)

# # codificar las variables categóricas
# dataset$Country = factor(dataset$Country,
#                     levels = c("France", "Spain", "Germany"),
#                     labels = c(1,2,3))
# dataset$Purchased = factor(dataset$Purchased,
#                         levels = c("No","Yes"),
#                         labels = c(0,1))

# Dividir los daos en conjunto de entrenamiento y conjunto de test
# install.packages("caTools")
library(caTools)
set.seet(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
print(split)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Escalado de valores
# training_set[,2:3] = scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3])