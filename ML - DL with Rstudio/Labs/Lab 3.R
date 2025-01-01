library(dplyr)
library(lubridate)
library(ggplot2)

############################

df <- read.csv("C:\\Users\\Nicho\\Desktop\\Superstore.csv")

############################

print(df$State[which(df$Sales == max(df$Sales))])

print(nrow(df[df$Order.Quantity > 50, ]))

print(length(unique(df$Item)))

print(IQR(df$Unit.Price))

print(round(mean(df[df$State == "Arizona", ]$Sales), 2))

df$Order.Date <- mdy(df$Order.Date)
avg_perdate <- df %>%
  group_by(Order.Date) %>%
  summarize(AverageSales = mean(Sales))
highest <- avg_perdate[which.max(avg_perdate$AverageSales), ]
formatted_date <- format(as.Date(highest$Order.Date), "%m%d%Y")
print(formatted_date)

print(df[which.max(with(df, Shipping.Cost / Sales)), ]$Item.ID)

avg_ship_cost <- df %>%
  group_by(Item.ID, Ship.Mode) %>%
  summarize(AverageShippingCost = mean(Shipping.Cost)) %>%
  ungroup()
del_truck_shipment <- filter(avg_ship_cost, Ship.Mode == "Delivery Truck")
highest_shipping_cost_item <- del_truck_shipment[which.max(del_truck_shipment$AverageShippingCost), ]
id <- highest_shipping_cost_item$Item.ID
print(id)


superstore_data <- df %>%
  mutate(Order.Date = as.Date(Order.Date, format = "%m/%d/%Y"))
paper_sales <- superstore_data %>%
  filter(Category == "Paper",
         Order.Date >= as.Date('2015-11-30') & Order.Date <= as.Date('2015-12-05'))
daily_sales <- paper_sales %>%
  group_by(Order.Date) %>%
  summarize(TotalSales = sum(Sales))
ggplot(daily_sales, aes(x = Order.Date, y = TotalSales)) +
  geom_line() +
  xlim(as.Date('2015-11-30'), as.Date('2015-12-05')) +
  labs(title = "Total Sales of Paper Category from 2015-11-30 to 2015-12-05",
       x = "Order Date",
       y = "Total Sales")
lowest_sales_date <- daily_sales[which.min(daily_sales$TotalSales), ]
formatted_lowest_sales_date <- format(as.Date(lowest_sales_date$Order.Date), "%m%d%Y")
print(formatted_lowest_sales_date)


superstore_data <- df %>%
  mutate(Order.Date = as.Date(Order.Date, format = "%m/%d/%Y"))
filtered_sales <- superstore_data %>%
  filter(Category %in% c('Binders and Binder Accessories', 'Telephones and Communication', 'Computer Peripherals'),
         Order.Date >= as.Date('2015-08-15') & Order.Date <= as.Date('2015-08-21'))
daily_sales <- filtered_sales %>%
  group_by(Order.Date, Category) %>%
  summarize(TotalSales = sum(Sales))
ggplot(daily_sales, aes(x = Order.Date, y = TotalSales, col = Category)) +
  geom_line() +
  xlim(as.Date('2015-08-15'), as.Date('2015-08-21')) +
  labs(title = "Total Sales from 2015-08-15 to 2015-08-21",
       x = "Order Date",
       y = "Total Sales")
sales_on_20150821 <- daily_sales[daily_sales$Order.Date == as.Date('2015-08-21'), ]
highest_sales_category <- sales_on_20150821[which.max(sales_on_20150821$TotalSales), ]$Category
print(highest_sales_category)


superstore_data <- df %>%
  mutate(Order.Date = ymd(Order.Date),
         Ship.Date = mdy(Ship.Date))
sum(is.na(df$Order.Date))
sum(is.na(df$Ship.Date))
superstore_data <- df %>%
  mutate(Lead.Time = difftime(Ship.Date, Order.Date, units = "days"))
average_lead_time <- df %>%
  group_by(`Item ID`) %>%
  summarize(AverageLeadTime = mean(Lead.Time, na.rm = TRUE))
longest_lead_time_item <- average_lead_time[which.max(average_lead_time$AverageLeadTime), ]
longest_lead_time_days <- as.numeric(longest_lead_time_item$AverageLeadTime)
print(longest_lead_time_days)


df$Category <- as.factor(df$Category)
model <- lm(Sales ~ Unit.Price + Category + Discount, data= df)
coeff <- names(coef(model))
print(coeff)
unit_price_coefficient <- coef(model)[["Unit.Price"]]
rounded_coefficient <- round(unit_price_coefficient, 2)
print(rounded_coefficient)


item_level_data <- df %>%
  group_by(Item.ID) %>%
  summarize(TotalSales = sum(Sales),
            AverageUnitPrice = mean(Unit.Price),
            AverageDiscount = mean(Discount),
            AverageShippingCost = mean(Shipping.Cost))
df$Category <- as.factor(df$Category)
item_level_data <- merge(item_level_data, df[, c("Item.ID", "Category")], by = "Item.ID")
model <- lm(TotalSales ~ AverageUnitPrice + AverageDiscount + AverageShippingCost + Category, data= item_level_data)
coeff <- names(coef(model))
print(coeff)
unit_price_coefficient <- coef(model)["AverageUnitPrice"]
rounded_coefficient <- round(unit_price_coefficient, 2)
print(rounded_coefficient)


df$Positive.Profit <- ifelse(df$Profit > 0, 1, 0)
logistic_model <- glm(Positive.Profit ~ Sales + Shipping.Cost + Discount + Unit.Price, data= df, family = binomial())
discount_coefficient <- coef(logistic_model)["Discount"]
rounded_coefficient <- round(discount_coefficient, 2)
print(rounded_coefficient)


df$High.Sales <- ifelse(df$Sales > 100, 1, 0)
df$Category <- as.factor(df$Category)
df$State <- as.factor(df$State)
probit_model <- glm(High.Sales ~ Category + Discount + Order.Quantity + Product.Base.Margin + State + Shipping.Cost + Unit.Price, data= df, family = binomial(link = "probit"))
discount_coefficient <- coef(probit_model)["Discount"]
rounded_coefficient <- round(discount_coefficient, 2)
print(rounded_coefficient)










