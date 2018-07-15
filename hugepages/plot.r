library("ggplot2")
library("scales")

get_data <- function(file) {
  x = read.csv(file, header=F)
  res = aggregate(x[,-1], list(x[,1]), mean)

  names(res) <- c("slots", "ipc", "missratio", "time")
  res <- transform(res, normtime = slots / time)
}

fast <- get_data(filefast)
slow <- get_data(fileslow)
slow$slots <- NULL
names(slow) <- c("ipcslow", "missratioslow", "timeslow", "normtimeslow")
res <- cbind(fast, slow)

ggplot(res, aes(x = slots, y = normtime, colour = "Optimizado")) +
  geom_line() +
  geom_point() +
  geom_line(aes(y = normtimeslow, colour = "Original")) +
  geom_point(aes(y = normtimeslow, colour = "Original")) +
  scale_x_log10(labels = comma) +
  labs(y = "Tiempo de ejecución normalizado (slots/s)") +
  labs(x = "Tamaño del problema (slots, log)") +
  theme(legend.title=element_blank()) +
  geom_vline(xintercept=8192) + # L1
  geom_vline(xintercept=65536) + # L2
  geom_vline(xintercept=3932160) # L3

ggplot(res, aes(x = slots, y = ipc, colour = "Optimizado")) +
  geom_line() +
  geom_point() +
  geom_line(aes(y = ipcslow, colour = "Original")) +
  geom_point(aes(y = ipcslow, colour = "Original")) +
  scale_x_log10(labels = comma) +
  labs(y = "Instrucciones por ciclo (IPC)") +
  labs(x = "Tamaño del problema (slots, log)") +
  theme(legend.title=element_blank()) +
  geom_vline(xintercept=8192) + # L1
  geom_vline(xintercept=65536) + # L2
  geom_vline(xintercept=3932160) # L3

ggplot(res, aes(x = slots, y = missratio, colour = "Optimizado")) +
  geom_line() +
  geom_point() +
  geom_line(aes(y = missratioslow, colour = "Original")) +
  geom_point(aes(y = missratioslow, colour = "Original")) +
  scale_x_log10(labels = comma) +
  labs(y = "Cache miss ratio (%)") +
  labs(x = "Tamaño del problema (slots, log)") +
  theme(legend.title=element_blank()) +
  geom_vline(xintercept=8192) + # L1
  geom_vline(xintercept=65536) + # L2
  geom_vline(xintercept=3932160) # L3
