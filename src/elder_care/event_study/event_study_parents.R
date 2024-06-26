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
library(coefplot)
library(ggfixest)

BAD <- 1
FEMALE <- 2


# Load data
data <- read.csv('/home/sebastian/Projects/elder_care/bld/event_study/sandbox_parents.csv')
#data <- subset(data, distance_to_treat >= -6 & distance_to_treat <= 12)

table(data$int_year)


table(data$informal_care_child)
# 0    1
# 3609  921

table(data$informal_care_daughter)



dat_raw <- as.data.table(data, TRUE)
is.data.table(dat_raw)

table(dat_raw$distance_to_treat)


# Step 2: Drop rows where distance_to_treat is -1, -3, -5, 1, 3, 5
dat_raw <- dat_raw %>%
  filter(!binned_distance_to_treat %in% c(-1, -3, -5, -7, -9, 1, 3, 5, 7, 9))

mergeids_to_drop <- dat_raw %>%
  filter(distance_to_treat == 0 & age < 75) %>%
  select(mergeid) %>%
  distinct()

dat <- dat_raw %>%
  anti_join(mergeids_to_drop, by = "mergeid")


table(dat$informal_care_daughter)


mergeids_to_drop_2 <- dat %>%
  filter(treat_ever == 0 & health == BAD) %>%
  select(mergeid) %>%
  distinct()

dat <- dat %>%
  anti_join(mergeids_to_drop_2, by = "mergeid")


table(dat$informal_care_daughter)
table(dat$informal_care_son)

table(dat$informal_care_daughter_and_other_child)
table(dat$informal_care_son_and_other_child)

#dat <- dat %>%
#  group_by(mergeid) %>%
#  mutate(distance_to_treat = if_else(treat_ever == 0, NA_real_, distance_to_treat)) %>%
#  ungroup()
dat <- dat %>%
  group_by(mergeid) %>%
  mutate(year_treated = int_year[distance_to_treat == 0][1]) %>%
  ungroup()

table(dat$year_treated, dat$distance_to_treat)
table(dat$int_year)

x_min_value <- -10
x_max_value <- 10

y_min_value <- -0.2
y_max_value <- 700

dat_all <- subset(dat, age >= 68)
dat_mother <- subset(dat, age >= 68 & gender == FEMALE)


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



# Control group: Parent never in bad health
mod_twfe_all = feols(informal_care_child ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                     |
                       #age ,                             ## FEs
                       mergeid + age,                         ## FEs
                     cluster = ~mergeid,   ## Clustered SEs
                     weights = ~hh_weight,
                     data = dat_all
)
iplot(mod_twfe_all,
      xlab = 'time to event (mother or father in bad health)',
      ylab = 'informal care by child (daughter)',
      main = '',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_all)


mod_twfe_mother = feols(informal_care_child ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                        |
                          #age ,                              ## FEs
                          mergeid + age,                         ## FEs
                        cluster = ~mergeid,   ## Clustered SEs
                        weights = ~hh_weight,
                        data = dat_mother
)
iplot(mod_twfe_mother,
      xlab = 'time to event (mother in bad health)',
      ylab = 'informal care by child (daughter)',
      main = '',
      xaxt='n'
      )
axis(1,at=c(10,30,40))
summary(mod_twfe_mother)


############
# Daughter #
############
table(dat_all$informal_care_daughter, dat_all$binned_distance_to_treat)
table(dat_mother$informal_care_son, dat_mother$binned_distance_to_treat)



mod_twfe_all = feols(informal_care_daughter ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                          |
                            #age ,                             ## FEs
                           mergeid + age,                         ## FEs
                          cluster = ~mergeid,   ## Clustered SEs
                          weights = ~hh_weight,
                          data = dat_all
)
iplot(mod_twfe_all,
      xlab = 'Zeit zum Ereignis (in Jahren): Eltern in schlechtem Gesundheitszustand',
      ylab = 'Informelle Pflege durch Tochter',
      main = '',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_all)




mod_twfe_mother = feols(informal_care_daughter ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                        |
                          #age ,                              ## FEs
                          mergeid + age,                         ## FEs
                        cluster = ~mergeid,   ## Clustered SEs
                        weights = ~design_weight,
                        data = dat_mother
)
pdf("SampleGraph.pdf",width=10,height=10)
iplot(mod_twfe_mother,
      xlab = 'Zeit zum Ereignis (in Jahren): Mutter in schlechtem Gesundheitszustand',
      ylab = 'Informelle Pflege durch Tochter',
      main = '',
      x.lim = c(-100, 100),
      y.lim = c(-100, 100))
dev.off() # turn off plotting

summary(mod_twfe_mother)


# Define your x-axis limits
x_min_value <- -5  # Set to your desired minimum x value
x_max_value <- 5   # Set to your desired maximum x value

# Generate the event study plot
iplot(mod_twfe_mother,
      xlab = 'Zeit zum Ereignis (in Jahren): Mutter in schlechtem Gesundheitszustand',
      ylab = 'Informelle Pflege durch Tochter',
      main = '',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-0.1, 0.1))


tidy_mod <- broom::tidy(mod_twfe_mother, conf.int = TRUE)
event_study_coeffs <- tidy_mod[grepl("binned_distance_to_treat", tidy_mod$term), ]
event_study_coeffs$term <- as.numeric(gsub("binned_distance_to_treat::treat_ever::", "", event_study_coeffs$term))


x_min_value <- -5  # Set to your desired minimum x value
x_max_value <- 5   # Set to your desired maximum x value

ggplot(event_study_coeffs, aes(x = term, y = estimate)) +
  geom_point() +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.2) +
  theme_minimal() +
  labs(title = '',
       x = 'Zeit zum Ereignis (in Jahren): Mutter in schlechtem Gesundheitszustand',
       y = 'Informelle Pflege durch Tochter') +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  xlim(x_min_value, x_max_value) +
  ylim(-10000, 100000)


#######
# Son #
#######
table(dat_all$informal_care_son, dat_all$binned_distance_to_treat)


dat_all_son <- dat_all %>%
  filter(!binned_distance_to_treat %in% c(-8, 6))

mod_twfe_all = feols(informal_care_son ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                     |
                       #age ,                             ## FEs
                       mergeid + age,                         ## FEs
                     cluster = ~mergeid,   ## Clustered SEs
                     weights = ~hh_weight,
                     data = dat_all_son
)
iplot(mod_twfe_all,
      xlab = 'Zeit zum Ereignis (in Jahren): Eltern in schlechtem Gesundheitszustand',
      ylab = 'Informelle Pflege durch Sohn',
      main = '',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_all)

# too few obs
table(dat_mother$informal_care_son, dat_mother$binned_distance_to_treat)
dat_mother_son <- dat_mother %>%
  filter(!binned_distance_to_treat %in% c(-8, 6))

mod_twfe_mother = feols(informal_care_son ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                        |
                          #age ,                              ## FEs
                          mergeid + age,                         ## FEs
                        cluster = ~mergeid,   ## Clustered SEs
                        weights = ~design_weight,
                        data = dat_mother_son
)
iplot(mod_twfe_mother,
      xlab = 'Zeit zum Ereignis (in Jahren): Mutter in schlechtem Gesundheitszustand',
      ylab = 'Informelle Pflege durch Sohn',
      main = '',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_mother)


####################################
# With other child (unconditional) #
####################################


mod_twfe_all = feols(informal_care_daughter_and_other_child ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                     |
                       #age ,                             ## FEs
                       mergeid + age,                         ## FEs
                     cluster = ~mergeid,   ## Clustered SEs
                     weights = ~hh_weight,
                     data = dat_all
)
iplot(mod_twfe_all,
      xlab = 'time to event (mother or father in bad health)',
      ylab = 'informal care by daughter',
      main = '',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_all)


mod_twfe_mother = feols(informal_care_daughter_and_other_child ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                        |
                          #age ,                              ## FEs
                          mergeid + age,                         ## FEs
                        cluster = ~mergeid,   ## Clustered SEs
                        weights = ~design_weight,
                        data = dat_mother
)
iplot(mod_twfe_mother,
      xlab = 'time to event (mother in bad health)',
      ylab = 'informal care by daughter',
      main = '',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_mother)


mod_twfe_all = feols(informal_care_son_and_other_child ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                     |
                       #age ,                             ## FEs
                       mergeid + age,                         ## FEs
                     cluster = ~mergeid,   ## Clustered SEs
                     weights = ~hh_weight,
                     data = dat_all
)
iplot(mod_twfe_all,
      xlab = 'time to event (mother or father in bad health)',
      ylab = 'informal care by son',
      main = '',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_all)


mod_twfe_mother = feols(informal_care_son_and_other_child ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                        |
                          #age ,                              ## FEs
                          mergeid + age,                         ## FEs
                        cluster = ~mergeid,   ## Clustered SEs
                        weights = ~design_weight,
                        data = dat_mother
)
iplot(mod_twfe_mother,
      xlab = 'time to event (mother in bad health)',
      ylab = 'informal care by son',
      main = '',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_mother)

#########
# Sunab #
#########

# DOES NOT WORK HERE WITH DIFFERENTIAL YEAR BINNING

df = fread("https://raw.githubusercontent.com/LOST-STATS/LOST-STATS.github.io/master/Model_Estimation/Data/Event_Study_DiD/bacon_example.csv")

dat_all$yyear_treated <- copy(dat_all$year_treated)
setDT(dat_all)
dat_all[, year_treated := ifelse(treat_ever==0, 10000, yyear_treated)]
#dat_mother[, binned_distance_to_treat := ifelse(treat_ever==0, 10000, binned_distance_to_treat)]

mod_twfe_all = feols(informal_care_daughter ~ sunab(year_treated, int_year)
                     |
                       mergeid,                         ## FEs
                     cluster = ~mergeid,   ## Clustered SEs
                     weights = ~hh_weight,
                     data = dat_all
)
iplot(mod_twfe_all,
      xlab = 'time to event (mother or father in bad health)',
      ylab = 'informal care by daughter',
      main = '',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_all)
