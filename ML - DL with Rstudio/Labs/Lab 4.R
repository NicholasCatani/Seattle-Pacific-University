##### Libraries

library(dplyr)
library(readr)
library(tidyr)
library(tidyverse)
library(AER)
library(MASS)
library(truncreg)

##### CSV file

df <- read.csv("C:\\Users\\Nicho\\Desktop\\Superstore.csv")

##### Execution

df2015 <- df %>% 
  mutate(Order.Date = as.Date(Order.Date, format = "%m/%d/%Y")) %>%
  mutate(Year = as.numeric(format(Order.Date, "%Y"))) %>%
  filter(Year == 2015)
total_sales_by_state <- df2015 %>%
  group_by(State) %>%
  summarise(Total_Sales = sum(Sales)) %>%
  arrange(State)

if(nrow(total_sales_by_state) >= 30) {
  value_at_30_2 <- total_sales_by_state[30, 2]
  print(value_at_30_2)
} else {
  print("The table has fewer than 30 rows.")
}

#################################################################

df2016 <- df %>%
  mutate(Order.Date = as.Date(Order.Date, format = "%m/%d/%Y")) %>%
  mutate(Year = as.numeric(format(Order.Date, "%Y"))) %>%
  filter(Year == 2016)
total_sales_by_state_2016 <- df2016 %>%
  group_by(State) %>%
  summarise(Total_Sales = sum(Sales)) %>%
  arrange(State)

if(nrow(total_sales_by_state_2016) >= 35) {
  value_at_35_2 <- total_sales_by_state_2016[35, 2]
  print(value_at_35_2)
} else {
  print("The table has fewer than 35 rows.")
}

#################################################################

df5 <- bind_rows(df2015, df2016)

total_sales_by_state_and_year <- df5 %>%
  group_by(State, Year) %>%
  summarise(Total_Sales = sum(Sales, na.rm = TRUE)) %>%
  arrange(State, Year)

if(nrow(total_sales_by_state_and_year) >= 35) {
  value_35_2 <- total_sales_by_state_and_year[35, 3]
  print(value_35_2)
} else {
  print("The table has fewer than 35 rows.")
}

#################################################################

total_sales_by_state_and_year$Key <- "sales"
names(total_sales_by_state_and_year)[names(total_sales_by_state_and_year) == "Total_Sales"] <- "value"

sorted_table <- total_sales_by_state_and_year %>%
  arrange(State, Year)

if(nrow(sorted_table) >= 33) {
  value_at_33_3 <- sorted_table[33, 3]
  print(value_at_33_3)
} else {
  print("The table has fewer than 33 rows.")
}

#################################################################

order_quantity_by_state_and_year <- df5 %>%
  group_by(State, Year) %>%
  summarise(Order.Quantity = sum(Order.Quantity, na.rm = TRUE)) %>%
  mutate(Key = "order.quantity") %>%
  arrange(State, Year)

names(order_quantity_by_state_and_year)[names(order_quantity_by_state_and_year) == "Order.Quantity"] <- "value"
sorted_table2 <- order_quantity_by_state_and_year

if(nrow(sorted_table2) >= 90) {
  value_At_90_3 <- sorted_table2[90, 3]
} else {
  print("The table has fewer than 90 rows")
}

#################################################################

df6 <- bind_rows(sorted_table, sorted_table2)

sorted_df6 <- df6 %>%
  arrange(State, Year)

if(nrow(sorted_df6) >= 150) {
  value_at_150_3 <- sorted_df6[150, 3]
  print(value_at_150_3)
} else {
  print("The table has fewer than 150 rows.")
}

#################################################################

sales_by_date <- df %>%
  group_by(Order.Date) %>%
  summarise(Total_Sales = sum(Sales, na.rm = TRUE)) %>%
  ungroup()

complete_dates <- sales_by_date %>%
  complete(Order.Date = seq.Date(min(Order.Date), max(Order.Date), by = "day"), fill = list(Total_Sales = NA))

sorted_table3 <- arrange(complete_dates, Order.Date)

if(nrow(sorted_table3) >= 1000) {
  value_at_1000_2 <- sorted_table3[1000, 2]
  print(value_at_1000_2)
} else {
  print("The table has fewer than 1000 rows.")
}

##################################################################

filled_sales <- complete_dates %>%
  fill(Total_Sales, .direction = "up")

sorted_filled_table <- arrange(filled_sales, Order.Date)

if(nrow(sorted_filled_table) >= 890) {
  value_at_890_2 <- sorted_filled_table[890, 2]
  print(value_at_890_2)
} else {
  print("The table has fewer than 890 rows.")
}

##################################################################

filled_sales <- complete_dates %>%
  fill(Total_Sales, .direction = "down")

sorted_filled_table <- arrange(filled_sales, Order.Date)

if(nrow(sorted_filled_table) >= 890) {
  value_at_890_2 <- sorted_filled_table[890, 2]
  print(value_at_890_2)
} else {
  print("The table has fewer than 890 rows.")
}

##################################################################

df$Category <- as.factor(df$Category)

poisson_model <- glm(Order.Quantity ~ Unit.Price + Discount + Category, data = df, family = poisson())
summary_model <- summary(poisson_model)
discount_coefficient <- summary_model$coefficients["Discount", "Estimate"]
round_discount_coefficient <- round(discount_coefficient, 3)
print(round_discount_coefficient)

##################################################################

disp_test <- dispersiontest(poisson_model, trafo = 1)

if(disp_test$p.value < 0.05) {
  cat("p-value < 0.05; Overdispersion exists\n")
} else {
  cat("p-value > 0.05; Overdispersion does not exist\n")
}

##################################################################

california <- df %>%
  filter(State == "California", !is.na(Order.Priority), Order.Priority != "") %>%
  mutate(Order.Priority = factor(Order.Priority, levels = c("Low", "Medium", "High", "Critical")))

ordinal_model <- polr(Order.Priority ~ Sales + Order.Quantity + Shipping.Cost + Category, data = california, Hess=TRUE)
order_quantity_coefficient <- coef(ordinal_model)["Order.Quantity"]
rounded_coefficient <- round(order_quantity_coefficient, 3)
print(rounded_coefficient)

##################################################################

filtered_df <- df %>%
  filter(Order.Quantity <= 150)
filtered_df$Category <- as.factor(filtered_df$Category)
truncated_model <- truncreg(Order.Quantity ~ Unit.Price + Shipping.Cost + Discount + Category,
                            data = filtered_df,
                            point = 0,
                            direction = "left")
unit_price_coefficient <- coef(truncated_model)["Unit.Price"]
rounded_coefficient <- round(unit_price_coefficient, 3)
print(rounded_coefficient)
  
  
  
  
  
  
  
