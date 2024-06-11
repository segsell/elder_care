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
data <- read.csv('/home/sebastian/Projects/elder_care/bld/event_study/sandbox.csv')
dat <- as.data.table(data, TRUE)
is.data.table(dat)

dat_female_raw <-subset(dat, gender == 2)
dat_male_raw <-subset(dat, gender == 1)

# Step 2: Drop rows where distance_to_treat is -1, -3, -5, 1, 3, 5
dat_female <- dat_female_raw %>%
  filter(!distance_to_treat %in% c(-1, -3, -5, 1, 3, 5))
dat_male <- dat_male_raw %>%
  filter(!distance_to_treat %in% c(-1, -3, -5, 1, 3, 5))


#dat<-subset(data, time_to_treat > -15 & time_to_treat < 15)
#dat_female<-subset(dat_female, binned_distance_to_treat > -7 & binned_distance_to_treat < 7)
#dat_male<-subset(dat_male, binned_distance_to_treat > -7 & binned_distance_to_treat < 7)

to_drop <- dat_female %>%
  filter(intensive_care_general == TRUE & intensive_care_no_other == FALSE) %>%
  select(mergeid)
dat_female_drop <- dat_female %>%
  anti_join(to_drop, by = "mergeid")

dat_female_drop <- dat_female_drop %>%
  mutate(intensive_care_no_other = case_when(
    (intensive_care_general == FALSE & intensive_care_no_other == FALSE) ~ 0,
    is.na(intensive_care_general) | is.na(intensive_care_no_other) ~ 0,
    TRUE ~ intensive_care_no_other
  ))


to_drop_male <- dat_male %>%
  filter(intensive_care_general == TRUE & intensive_care_no_other == FALSE) %>%
  select(mergeid)
dat_male_drop <- dat_male %>%
  anti_join(to_drop_male, by = "mergeid")

dat_male_drop <- dat_male_drop %>%
  mutate(intensive_care_no_other = case_when(
    (intensive_care_general == FALSE & intensive_care_no_other == FALSE) ~ 0,
    is.na(intensive_care_general) | is.na(intensive_care_no_other) ~ 0,
    TRUE ~ intensive_care_no_other
  ))


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

mod_twfe_ft_women = feols(intensive_care_no_other ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                 |
                #age ,                             ## FEs
                mergeid + age + int_year,                         ## FEs
                 cluster = ~mergeid,   ## Clustered SEs
                 #weights = ~design_weight,
                 data = dat_female_drop
)
iplot(mod_twfe_ft_women,
      xlab = 'time to treatment',
      main = 'Working Full Time: Women',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_ft_women)


mod_twfe_ft_men = feols(intensive_care_no_other ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                          |
                            #age ,                             ## FEs
                            mergeid + age + int_year,                         ## FEs
                          cluster = ~mergeid,   ## Clustered SEs
                          #weights = ~design_weight,
                          data = dat_male_drop
)
iplot(mod_twfe_ft_men,
      xlab = 'time to treatment',
      main = 'Working Full Time: Men',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_ft_men)


mod_twfe_ft_women = feols(full_time ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                          |
                            #age ,                             ## FEs
                            mergeid + age + int_year,                         ## FEs
                          cluster = ~mergeid,   ## Clustered SEs
                          #weights = ~design_weight,
                          data = dat_female
)
iplot(mod_twfe_ft_women,
      xlab = 'time to treatment',
      main = 'Working Full Time: Women',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_ft_women)


mod_twfe_ft_men = feols(full_time ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                        |
                          #age ,                             ## FEs
                          mergeid + age + int_year,                         ## FEs
                        cluster = ~mergeid,   ## Clustered SEs
                        #weights = ~design_weight,
                        data = dat_male
)
iplot(mod_twfe_ft_men,
      xlab = 'time to treatment',
      main = 'Working Full Time: Men',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_ft_men)
