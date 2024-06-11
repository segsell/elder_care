#######################
# Sub-sample Analysis #
#######################
library(foreign)
library(data.table) ## For some minor data wrangling
library(haven)     ## NB: Requires version >=0.9.0
library(fixest)
library(AER)
library(did)
library(lfe)
library(dplyr)


# Load data
data <- read.csv('/home/sebastian/Projects/elder_care/bld/event_study/sandbox_parents.csv')
dat_raw <- as.data.table(data, TRUE)
is.data.table(dat_raw)


# Step 2: Drop rows where distance_to_treat is -1, -3, -5, 1, 3, 5
dat <- dat_raw %>%
  filter(!distance_to_treat %in% c(-1, -3, -5, 1, 3, 5))

dat <- subset(dat_raw, age >= 75)
dat <- subset(dat_raw, age >= 75 & gender == 2)


x_min_value <- -10
x_max_value <- 10

y_min_value <- -0.2
y_max_value <- 700


######################
# Outcome: #
######################

# sub <- subset(dat_female, age >= 40 & age <= 70)
# sub <- sub %>% select(pid, syear, current_part_time, care_ever, time_to_treat, age)
# sub <- sub %>% arrange(pid, syear)

# selected_pids <- sub %>%
#  filter(age >= 50 + 2 & age <= 60 -2 & time_to_treat == 0) %>%
#  distinct(pid)

# dat_est <- sub %>%
#  filter(pid %in% selected_pids$pid)

# ep013_

mod_twfe = feols(informal_care_child ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                          |
                            #age ,                             ## FEs
                           mergeid + age,                         ## FEs
                          cluster = ~mergeid,   ## Clustered SEs
                          weights = ~hh_weight,
                          data = dat
)
iplot(mod_twfe,
      xlab = 'time to treatment',
      main = 'Working Full Time: Women',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe)

