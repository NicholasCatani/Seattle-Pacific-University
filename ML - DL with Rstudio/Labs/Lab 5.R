######################## Libraries

library(nycflights13)
library(dplyr)
library(survival)
library(survminer)
library(cmprsk)

######################## Datasets

df <- read.csv("C:\\Users\\Nicho\\Desktop\\lung_disease.csv")
df2 <- nycflights13::flights
df3 <- nycflights13::airlines
df4 <- nycflights13::planes
df5 <- nycflights13::weather
df6 <- nycflights13::airports

######################## Questions

merged_df <- merge(df2, df3, on="carrier")
print(virgin_american <- merged_df %>%
  filter(name == "Virgin America") %>%
  nrow())

merged_df2 <- merge(df2, df4, on="tailnum")
print(embraer <- merged_df2 %>%
      filter(manufacturer == "EMBRAER") %>%
      nrow())

merged_df3 <- merge(df2, df5, on= c("year", "month", "day", "hour", "origin"))
print(wind_speed <- merged_df3 %>%
      filter(wind_speed > 11) %>%
      nrow())

merged_df4 <- df2 %>%
  inner_join(df6, by = c("origin" = "faa"))
print(latitude <- merged_df4 %>%
      filter(lat > 40.7) %>%
      nrow())

aa_code  <- df3 %>%
  filter(grepl("American Airlines", name)) %>%
  pull(carrier)
aa_flights_to_lga <- df2 %>%
  filter(dest == "LGA", carrier %in% aa_code)
print(nrow(aa_flights_to_lga))

embraer_flights <- merged_df2 %>%
  filter(manufacturer == "EMBRAER")
print(sum(embraer_flights$seats, na.rm = TRUE))

emb145 <- merged_df2 %>%
  filter(model == "EMB-145XR")
print(sum(emb145$distance, na.rm = TRUE))

merged_df5 <- merge(merged_df2, df3, on="carrier")
man_owed <- merged_df5 %>%
  filter(manufacturer == "EMBRAER" & name == "JetBlue Airways")
print(nrow(man_owed))

envoy_air <- merged_df %>%
  filter(name == "Envoy Air" & air_time > 150)
print(nrow(envoy_air))

non <- merged_df2 %>%
  filter(model == "")
print(nrow(non))

###################################################################################

surv_object <- Surv(time = df$time, event = df$status)
fit <- survfit(surv_object ~ ph.ecog, data = df)
ggsurvplot(fit, data = df, pval = TRUE, conf.int = TRUE,
           legend.title = "ph.ecog Score", legend.labs = c("0", "1", "2", "3"),
           xlab = "Time", ylab = "Survival Probability")

cum_incidence <- cuminc(ftime = df$time, fstat = df$status, group = df$ph.ecog)
plot(cum_incidence, mark.time = TRUE, xlab = "Time", ylab = "Cumulative Incidence")

log_rank_result <- survdiff(surv_object ~ ph.ecog, data = df, rho = 0)
print(round(log_rank_result$chisq))

cox_model2 <- coxph(Surv(time, status) ~ age + sex + ph.ecog + ph.karno + pat.karno + meal.cal + wt.loss, data = df)
summary_cox <- summary(cox_model2)
print(round(summary_cox$coefficients["ph.ecog", "Pr(>|z|)"], 3))
