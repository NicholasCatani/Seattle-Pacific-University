###################
## TEXT ANALYSIS ##
###################


####### STEP 1 ### DATA EXPLORATION


## LOADING THE CSV FILE AND CHECK THE DATASET

dataset <- read.csv("C:/Users/Nicho/Desktop/Seattle Pacific University/Data Mining/Text mining/All-seasons.csv")
head(dataset)


## CHECK HOW MANY CHARACTERS ARE IN THE SET

length(unique(dataset$Character))


## TOP 10 CHARACTERS BY NUMBER OF LINES SAID

library(dplyr)
library(magrittr)
dataset %>%
  count(Character) %>%
  arrange(desc(n)) %>%
  top_n(10)


## NUMBER OF SEASONS AND EPISODES

dataset %>%
  count(Season, Episode)



####### STEP 2 ### DATA PROCESSING


## PROPORTION OF POSITIVE AND NEGATIVE WORDS IN THE DATASET

library(tidytext)

clean_ds <- dataset %>%
  unnest_tokens(word, Line) %>%
  inner_join(get_sentiments("bing")) %>%
  anti_join(stop_words, by = "word")

clean_ds %>%
  count(sentiment)


## MOST POSITIVE AND NEGATIVE WORDS

lexic_ds <- clean_ds %>%
  count(word, sentiment)

top_20 <- lexic_ds %>%
  group_by(sentiment) %>%
  top_n(20) %>%
  ungroup() %>%
  mutate(word = reorder(word, n))

library(ggplot2)
ggplot(top_20, aes(word, n, fill = sentiment)) +
  geom_col() +
  facet_wrap(~sentiment, scales = "free") + coord_flip()


## ALTERNATE VERSION

library(reshape2)
library(wordcloud)

lexic_ds %>%
  acast(word ~ sentiment, value.var = "n", fill = 0) %>%
  comparison.cloud(max.words = 70)


## FINAL BAR CHART

count_words <- clean_ds %>%
  count(word, sentiment, sort = TRUE)

top_25 <- head(count_words, 25)
ggplot(top_25, aes(x = reorder(word, n), n, fill = sentiment)) + geom_col() + coord_flip()



####### STEP 3 ### FINAL RESULTS


## CHARACTER ANALYSIS

clean_ds$Character <- sapply(clean_ds$Character, tolower)

clean_ds %>%
  count(Character, sentiment) %>%
  arrange(desc(n)) %>%
  top_n(100)


## MAIN CHARACTERS COMPARISON


# Negative words

char_neg_words <- clean_ds %>%
  inner_join(get_sentiments("bing")) %>%
  filter(Character %in% c("stan", "kenny", "cartman", "kyle")) %>%
  filter(sentiment == "negative") %>%
  count(word, Character) %>%
  group_by(Character) %>%
  top_n(5) %>%
  ungroup() %>%
  mutate(List = reorder(paste(word), n))

ggplot(char_neg_words, aes(List, n)) + geom_col(show.legend = FALSE) +
  facet_wrap(~Character, nrow = 2, scales = "free") + coord_flip()


# Positive words

char_pos_words <- clean_ds %>%
  inner_join(get_sentiments("bing")) %>%
  filter(Character %in% c("stan", "kenny", "cartman", "kyle")) %>%
  filter(sentiment == "positive") %>%
  count(word, Character) %>%
  group_by(Character) %>%
  top_n(5) %>%
  ungroup() %>%
  mutate(List = reorder(paste(word), n))

ggplot(char_pos_words, aes(List, n)) + geom_col(show.legend = FALSE) +
  facet_wrap(~Character, nrow = 2, scales = "free") + coord_flip()


# SENTIMENT WORDS GROUPED BY SEASON

season <- clean_ds %>%
  inner_join(get_sentiments("bing")) %>%
  group_by(Season, sentiment) %>%
  count(Season, sentiment)

ggplot(season, aes(Season, n, fill - sentiment)) + geom_col() + facet_grid(~sentiment) + coord_flip()