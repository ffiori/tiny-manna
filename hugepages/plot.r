library("ggplot2")
library("scales")
x = read.csv(file, header=F)
res = aggregate(x[,-1], list(x[,1]), mean)

names(res) <- c("slots", "ipc", "missratio", "time")
res <- transform(res, normtime = slots / time)

ggplot(res, aes(x = slots, y = normtime)) +
  geom_line() +
  geom_point() +
  scale_x_log10(labels = comma) +
  labs(y = "Tiempo de ejecución normalizado (slots/s)") +
  labs(x = "Tamaño del problema (slots, log)") +
  theme(legend.title=element_blank()) + 
  geom_vline(xintercept=8192) + geom_vline(xintercept=65536) + geom_vline(xintercept=3932160)

ggplot(res, aes(x = slots, y = ipc)) +
  geom_line() +
  geom_point() +
  scale_x_log10(labels = comma) +
  labs(y = "Instrucciones por ciclo (IPC)") +
  labs(x = "Tamaño del problema (slots, log)") +
  theme(legend.title=element_blank()) +
  geom_vline(xintercept=8192) + geom_vline(xintercept=65536) + geom_vline(xintercept=3932160)

ggplot(res, aes(x = slots, y = missratio)) +
  geom_line() +
  geom_point() +
  scale_x_log10(labels = comma) +
  labs(y = "Cache miss ratio (%)") +
  labs(x = "Tamaño del problema (slots, log)") +
  theme(legend.title=element_blank()) +
  geom_vline(xintercept=8192) + geom_vline(xintercept=65536) + geom_vline(xintercept=3932160)
