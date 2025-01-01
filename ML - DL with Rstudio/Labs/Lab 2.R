library(ggplot2)
df <- read.csv("C:\\Users\\Nicho\\Desktop\\Superstore.csv")

#####################################

ggplot(df, aes(x = Shipping.Cost)) +
  geom_histogram(binwidth = 1) +
  theme_minimal() +
  labs(title = "Histogram of Shipping Costs", x = "Shipping Cost", y = "Frequency")


ggplot(df, aes(x = Order.Quantity)) +
  geom_histogram(bidwidth = 1) +
  facet_wrap(~Ship.Mode, scales = "free_y") +
  theme_minimal() +
  labs(title = "Histogram of Order Quantity by Ship Mode", x = "Order Quantity", y = "Frequency")


ggplot(df, aes(x = Unit.Price)) +
  geom_histogram(binwidth = 1) +
  coord_cartesian(xlim = c(1, 10)) +
  theme_minimal() +
  labs(title = " Histogram of Unit Price", x = "Unit Price", y = "Frequency")


ggplot(df, aes(x = Category, y = Unit.Price)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Boxplot of Unit Price by Product Category", x = "Product Category", y = "Unit Price")


ggplot(df, aes(fill = "Ship Mode", x = "Customer Segment")) +
  geom_bar(position = "stack") +
  theme_minimal() +
  labs(title = "Count of Transactions by Customer Segment and Ship Mode", x = "Customer Segment", y = "Count of Transactions")


ggplot(df, aes(x = Category, y = Sales, fill = Region)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  theme_minimal() +
  labs(title = "Sales by Product Category for Each Region", x = "Product Category", y = "Sales")


ggplot(df, aes(fill = `Customer Segment`, x = State)) +
  geom_bar(position = "stack") +
  theme_minimal() +
  labs(title = "Count of Customers by State and Customer Segment", x = "State", y = "Count of Customers")


df$Order.Date <- as.Date(df$Order.Date, "%m/%d/%Y")
df$year <- format(df$Order.Date, "%Y")
df$month <- format(df$Order.Date, "%m")
ggplot(df, aes(x = month, y = Sales)) +
  geom_boxplot() +
  coord_cartesian(ylim = c(1, 10)) + 
  theme_minimal() +
  labs(title = "Boxplot of Sales by Month", x = "Month", y = "Sales")


df$Order.Date <- as.Date(df$Order.Date, "%m/%d/%Y")
df$year_month <- format(df$Order.Date, "%Y/%m")
ggplot(df, aes(x = year_month, y = `Order Quantity`)) +
  geom_boxplot() +
  coord_cartesian(ylim = c(1, 10)) +
  theme_minimal() +
  labs(title = "Boxplot of Order Quantity by Year_Month", x = "Year_Month", y = "Order Quantity")


ggplot(data, aes(fill = `Customer Segment`, x = Region)) +
  geom_bar(position = "stack") +
  theme_minimal() +
  labs(title = "Total Number of Customers by Customer Segment and Region", x = "Region", y = "Number of Customers")


df$Order.Date <- as.Date(df$Order.Date, "%m/%d/%Y")
df$year_month <- format(df$Order.Date, "%Y/%m")
aggregated_data <- aggregate(`Order Quantity` ~ year_month, df, sum)
ggplot(aggregated_data, aes(x = year_month, y = `Order Quantity`)) +
  geom_bar(stat = "identity") +
  coord_cartesian(ylim = c(1, 10)) + 
  theme_minimal() +
  labs(title = "Bar Chart of Order Quantity by Year_Month", x = "Year_Month", y = "Order Quantity")


ggplot(df, aes(x = Profit, fill = Category)) +
  geom_density(alpha = 0.5) +
  theme_minimal() +
  labs(title = "Density Plot of Profit by Product Category", x = "Profit", y = "Density")


data_appliances <- subset(df, Category == 'Appliances', select = 'Unit.Price')
data_storage <- subset(df, Category == 'Storage & Organization', select = 'Unit.Price')
t_test_result <- t.test(data_appliances$`Unit.Price`, data_storage$`Unit.Price`)
p_value <- round(t_test_result$p.value, 2)
p_value


t_test_result <- t.test(df$`Unit Price`, mu = 83, alternative = "greater")
p_value <- round(t_test_result$p.value, 2)
p_value



model <- aov(Sales ~ Category + State + Category:State, data = df)
summary(model)
