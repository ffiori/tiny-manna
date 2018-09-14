library(ggplot2)
library(scales)

options(scipen=10000)

data = read.csv("melted.csv", header=T)
data$what = factor(data$what)

plot_field <- function(name, label) {
  filtered <- subset(data, var==name)

  p <- ggplot(filtered, aes(x=slots, y=value, group=what, color=what)) +
    geom_line() +
    geom_point() +
    scale_x_log10(labels = comma) +
    labs(y = label) +
    labs(x = "Tamaño del problema (slots, log)") +
    labs(color = "Método", linetype = "Método") +
    geom_vline(xintercept=8192) + # L1
    geom_vline(xintercept=65536) + # L2
    geom_vline(xintercept=3932160) # L3

  print(p)
}

plot_field("normtime", "Tiempo de ejecución normalizado (slots/s)")
