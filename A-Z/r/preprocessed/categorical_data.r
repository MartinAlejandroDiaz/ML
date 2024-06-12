# Plantilla de preprocesado de datos
dataset = read.csv('../Data.csv')

# codificar las variables categóricas
dataset$Country = factor(dataset$Country,
                    levels = c("France", "Spain", "Germany"),
                    labels = c(1,2,3))
dataset$Purchased = factor(dataset$Purchased,
                        levels = c("No","Yes"),
                        labels = c(0,1))
