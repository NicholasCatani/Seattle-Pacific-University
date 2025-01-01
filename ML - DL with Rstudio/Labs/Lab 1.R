seq <- seq(1, 500, by = 5)
mean_seq <- mean(seq)
rounded <- round(mean_seq, 1)
rounded

repeated_sequence <- rep(c(1, 2, 3), times = 10)
variance_sequence <- var(repeated_sequence)
rounded_variance <- round(variance_sequence, 2)
rounded_variance

repeated_sequence <- rep(c(1, 2, 3), times = 10)
repeated_sequence_2 <- rep(c(4, 5, 6), times = 10)
concatenation <- c(repeated_sequence, repeated_sequence_2)
std <- sd(concatenation)
rounded <- round(std, 2)
rounded

seq <- seq(1, 500, by = 5.5)
mean_seq <- quantile(seq)
rounded <- round(mean_seq, 2)
rounded



df <- read.csv("C:\\Users\\Nicho\\Desktop\\Superstore.csv")

mean_sales <- mean(df$Sales)
rounded_mean_sales <- round(mean_sales, 1)
rounded_mean_sales

df[500, 21]

df$Sales[300]

df[655, 15] - df[815, 21]

df[860, 15] == df[54, 24]

round(cor(df$Profit, df$Unit.Price), 2)

round(mean(df[, 15]), 1)

sum(df$Profit < 0)

sum(df$Order.Quantity == 1)

length(unique(df$Sales))


profit_list <- df$Profit
value_at_index_16 <- profit_list[17]
ranked_profits <- rank(profit_list, ties.method = "min")
rank_of_value_at_index_16 <- ranked_profits[17]
rank_of_value_at_index_16
